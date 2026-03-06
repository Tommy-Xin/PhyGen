export AE="/home/gaiyiming/hjq/xinc/DiffusionCLIP/pretrained_weights/FLUX.1-dev/ae.safetensors"

export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --config_file "train_configs/accelerate_config.yaml" training/train_SigLIP_stage1.py --config "train_configs/test_SigLIP_384_stage1.yaml"
