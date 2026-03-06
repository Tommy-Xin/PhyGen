export AE="/home/gaiyiming/hjq/xinc/DiffusionCLIP/pretrained_weights/FLUX.1-dev/ae.safetensors"

export CUDA_VISIBLE_DEVICES=4,5,0,7
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --config_file "train_configs/accelerate_config.yaml" training/train_SigLIP_stage2_all_eval.py --config "train_configs/test_SigLIP_384_stage2_all_eval.yaml"
