from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor
import argparse
import sys

from einops import rearrange
import uuid
import os
from torchvision import transforms
import matplotlib.pyplot as plt


from src.flux.sampling import denoise_controlnet, get_noise, get_schedule, prepare, unpack
from src.flux.model import Flux
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    get_lora_rank,
    load_checkpoint
)

from image_datasets.dataset_cc3m import image_transform
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5)

from clip_models.build_CLIP import load_clip_model_OpenAICLIP
from clip_models.sampling import prepare_clip
from torchvision import transforms



def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1.0,
):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale, 
        )
        if i >= timestep_to_start_cfg:
            pred =  true_gs * pred
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img



OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
VAE_MEAN = 0.5
VAE_STD = 0.5
NORMALIZE_CLIP = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
NORMALIZE_VAE = transforms.Normalize(mean=VAE_MEAN, std=VAE_STD)


class XFluxPipeline:
    def __init__(self, ae, model, clip_vis, device):
        self.offload = False

        self.ae = ae
        self.model = model
        self.model.eval()
        self.clip_vis = clip_vis
        self.seq_t5 = 512
        self.device = device

    def __call__(self,
                 original_img,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 ip_scale: float = 1.0,
                 timestep_to_start_cfg: int = 0,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)

        return self.forward(
            original_img,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image=None,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            image_proj=None,
            ip_scale=ip_scale,
        )

    def forward(
        self,
        original_img,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        image_proj=None,
        ip_scale=1.0,
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.clip_vis = self.clip_vis.to(self.device)

            #inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            inp_cond = prepare_clip(clip=self.clip_vis, seq_t5=self.seq_t5, original_img=NORMALIZE_CLIP(original_img).to(torch.bfloat16), img=x)

            if self.offload:
                self.model = self.model.to(self.device)

            x = denoise(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                true_gs=true_gs,
                image_proj=image_proj,
                ip_scale=ip_scale,
            )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        return x

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()


def save_image(image, filename):
    image = image * VAE_STD + VAE_MEAN
    image = rearrange(image[-1], "c h w -> h w c").cpu().numpy()
    sizes = image.shape
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    plt.savefig(filename, dpi = sizes[0]) 
    plt.close()


def save_image(image, filename):
    image = image * VAE_STD + VAE_MEAN
    image = rearrange(image[-1], "c h w -> h w c")
    image = (image-image.min()) / (image.max()-image.min()) * 255
    image = Image.fromarray(image.cpu().byte().numpy())
    image.save(filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_image_size", type=int, default=224, help="The width for generated image")
    parser.add_argument("--clip_dim", type=int, default=768, help="The width for generated image")
    parser.add_argument("--t5_dim", type=int, default=4096, help="The width for generated image")
    args = parser.parse_args()

    model_name = "flux-dev"

    idx = 2
    load_dir = "output_mlp2_global_sigmoid_stage1_s1.0"
    load_step = 70000
    image_path = f'generated_images/test{idx}.jpg'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 224
    sample_steps = 20
    sample_width, sample_height = 224, 224

    train_processor = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size), transforms.ToTensor(),
        ])
    input_img = Image.open(image_path).convert('RGB')
    input_img = train_processor(input_img).to(device)
    input_img = input_img.unsqueeze(0)

    # save input img
    save_image(input_img, f'generated_images/input{idx}.jpg')

    dit = load_flow_model2(model_name, device="cpu")
    vae = load_ae(model_name, device=device)
    clip_vis = load_clip_model_OpenAICLIP(args, device=device)

    print('loading projection params...')
    load_path_project_clip = os.path.join(load_dir, f"checkpoint-project-clip-{load_step}.bin")
    clip_vis.project_clip.load_state_dict(torch.load(load_path_project_clip, map_location=torch.device('cpu')))
    load_path_project_t5 = os.path.join(load_dir, f"checkpoint-project-t5-{load_step}.bin")
    clip_vis.project_t5.load_state_dict(torch.load(load_path_project_t5, map_location=torch.device('cpu')))
    print('loading successfully!')

    print('loading dit params...')
    load_path_dit = os.path.join(load_dir, f"checkpoint-dit-{load_step}.bin")
    dit.load_state_dict(torch.load(load_path_dit, map_location=torch.device('cpu')))
    print('loading successfully!')

    clip_vis = clip_vis.to(torch.bfloat16)

    vae.requires_grad_(False)
    dit.requires_grad_(False)
    dit = dit.to(torch.bfloat16)
    dit.to(device)

    # generate and save output img
    sampler = XFluxPipeline(ae=vae, model=dit, clip_vis=clip_vis, device=device)
    output_img = sampler(input_img,
                 width=sample_width,
                 height=sample_height,
                 num_steps=sample_steps,
                 true_gs=1.0,   # NOTE!!!
                 )

    save_image(output_img, f'generated_images/output{idx}_stage1_load{load_step}.jpg')
