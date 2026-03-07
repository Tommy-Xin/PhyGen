export AE="/data2/xinc/PhyGen/Continuous/pretrained_weights/FLUX.1-dev/ae.safetensors"

export CUDA_VISIBLE_DEVICES=0,1,2,3
ROOT_DIR="/data2/xinc/PhyGen/Continuous"
cd "${ROOT_DIR}" || exit 1

accelerate launch --config_file "train_configs/accelerate_config.yaml" \
    "training/clip_loss/train_OpenAICLIP_stage1_clip_loss.py" \
    --config "train_configs/clip_loss/test_OpenAICLIP_336_stage1_clip_loss.yaml"
