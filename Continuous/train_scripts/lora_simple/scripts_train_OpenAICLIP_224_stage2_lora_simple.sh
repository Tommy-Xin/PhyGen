export AE="/home/gaiyiming/hjq/xinc/DiffusionCLIP/pretrained_weights/FLUX.1-dev/ae.safetensors"

export CUDA_VISIBLE_DEVICES=4,5,6,7
ROOT_DIR="/home/gaiyiming/hjq/xinc/GenHancer/Continuous"
cd "${ROOT_DIR}" || exit 1

accelerate launch --config_file "train_configs/accelerate_config.yaml" \
    "training/lora_simple/train_OpenAICLIP_stage2_lora_simple.py" \
    --config "train_configs/lora_simple/test_OpenAICLIP_224_stage2_lora_simple.yaml"
