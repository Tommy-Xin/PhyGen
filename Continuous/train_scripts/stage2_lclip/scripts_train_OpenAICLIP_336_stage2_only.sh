export AE="/data2/xinc/PhyGen/Continuous/pretrained_weights/FLUX.1-dev/ae.safetensors"

export CUDA_VISIBLE_DEVICES=0,1,2,3
ROOT_DIR="/data2/xinc/PhyGen/Continuous"
cd "${ROOT_DIR}" || exit 1

accelerate launch --config_file "train_configs/accelerate_config.yaml" \
    "training/stage2_lclip/train_OpenAICLIP_stage2_only.py" \
    --config "train_configs/stage2_lclip/test_OpenAICLIP_336_stage2_only.yaml"
