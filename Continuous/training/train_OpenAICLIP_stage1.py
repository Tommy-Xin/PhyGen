import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, unpack
from src.flux.util import configs, load_ae, load_flow_model2

from image_datasets.dataset_cc3m import loader
from torchvision import transforms

from clip_models.build_CLIP import load_clip_model_OpenAICLIP
from clip_models.sampling import prepare_clip

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")


OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
VAE_MEAN = 0.5
VAE_STD = 0.5
NORMALIZE_CLIP = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
NORMALIZE_VAE = transforms.Normalize(mean=VAE_MEAN, std=VAE_STD)


class SuperModel(nn.Module):
    def __init__(self, clip_vis, dit):
        super().__init__()
        self.clip_vis = clip_vis
        self.dit = dit
    
    def get_clip_vis(self):
        return self.clip_vis
    
    def get_dit(self):
        return self.dit


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config


def main():
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    args.clip_config.seq_t5 = 256 if is_schnell else 512   # NOTE!!!
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    dit = load_flow_model2(args.model_name, device="cpu")
    vae = load_ae(args.model_name, device=accelerator.device)
    clip_vis = load_clip_model_OpenAICLIP(args.clip_config, device=accelerator.device)


    # contiguous projections for OpenAICLIP-336px
    if args.clip_config.clip_image_size == 336:
        clip_vis.model.visual_projection.weight = torch.nn.Parameter(clip_vis.model.visual_projection.weight.contiguous())
        clip_vis.model.text_projection.weight = torch.nn.Parameter(clip_vis.model.text_projection.weight.contiguous())


    vae.requires_grad_(False)
    dit.requires_grad_(True)
    dit = dit.to(torch.bfloat16)
    dit.to(accelerator.device)
    clip_vis.train()
    dit.train()

    for name_, param in clip_vis.named_parameters():
        if 'project_clip' in name_ or 'project_t5' in name_:
            param.requires_grad = True
        else:
            param.requires_grad = False
    

    # super model = clip_vis + dit
    super_model = SuperModel(clip_vis, dit)

    optimizer = torch.optim.AdamW(
        [p for p in super_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(int(3e6) / args.data_config.train_batch_size) / args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    super_model, optimizer, _, lr_scheduler = accelerator.prepare(
        super_model, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # NOTE!!!   # accumulate super_model
            with accelerator.accumulate(super_model):
                original_img, prompts = batch['image'], batch['text']
                original_img = original_img.to(accelerator.device)
                
                with torch.no_grad():
                    x_1 = vae.encode(NORMALIZE_VAE(original_img).to(torch.float32))

                inp = prepare_clip(clip=super_model.clip_vis, original_img=NORMALIZE_CLIP(original_img).to(weight_dtype), img=x_1.to(weight_dtype))
                x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                bs = original_img.shape[0]
                t = torch.sigmoid(torch.randn((bs,), device=accelerator.device) * args.scale_factor)
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                # Predict the noise residual and compute loss
                model_pred = super_model.dit(img=x_t.to(weight_dtype),
                                             img_ids=inp['img_ids'].to(weight_dtype),
                                             txt=inp['txt'].to(weight_dtype),
                                             txt_ids=inp['txt_ids'].to(weight_dtype),
                                             y=inp['vec'].to(weight_dtype),
                                             timesteps=t.to(weight_dtype),
                                             guidance=guidance_vec.to(weight_dtype),)

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(super_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwrapped_super_model = accelerator.unwrap_model(super_model)
                        # save dit ckpts
                        save_path_dit = os.path.join(args.output_dir, f"checkpoint-dit-{global_step}.bin")
                        torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)
                        # save project clip ckpts
                        save_path_project_clip = os.path.join(args.output_dir, f"checkpoint-project-clip-{global_step}.bin")
                        torch.save(deepcopy(unwrapped_super_model.clip_vis.project_clip).state_dict(), save_path_project_clip)
                        # save project t5 ckpts
                        save_path_project_t5 = os.path.join(args.output_dir, f"checkpoint-project-t5-{global_step}.bin")
                        torch.save(deepcopy(unwrapped_super_model.clip_vis.project_t5).state_dict(), save_path_project_t5)
                        # save optimizer state
                        save_path_optimizer = os.path.join(args.output_dir, f"optimizer-state-{global_step}.bin")
                        torch.save(optimizer.state_dict(), save_path_optimizer)

                        logger.info(f"Saved ckpts to {save_path_dit}, {save_path_project_clip} and {save_path_project_t5}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    unwrapped_super_model = accelerator.unwrap_model(super_model)
                    # save dit ckpts
                    save_path_dit = os.path.join(args.output_dir, f"checkpoint-dit-{global_step}.bin")
                    torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)
                    # save project clip ckpts
                    save_path_project_clip = os.path.join(args.output_dir, f"checkpoint-project-clip-{global_step}.bin")
                    torch.save(deepcopy(unwrapped_super_model.clip_vis.project_clip).state_dict(), save_path_project_clip)
                    # save project t5 ckpts
                    save_path_project_t5 = os.path.join(args.output_dir, f"checkpoint-project-t5-{global_step}.bin")
                    torch.save(deepcopy(unwrapped_super_model.clip_vis.project_t5).state_dict(), save_path_project_t5) 

                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
