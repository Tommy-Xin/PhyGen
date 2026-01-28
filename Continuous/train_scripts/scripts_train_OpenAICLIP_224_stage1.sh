export AE="/home/gaiyiming/hjq/xinc/DiffusionCLIP/pretrained_weights/FLUX.1-dev/ae.safetensors"

# Specify available GPUs (e.g., use GPUs 0,1,2,3)
# Uncomment and modify the line below to specify which GPUs to use
export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_stage1.py --config "train_configs/test_OpenAICLIP_224_stage1.yaml"
