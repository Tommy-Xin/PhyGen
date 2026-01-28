# GenHancer with Continuous Generative Models



## 🔎 Introduction

* For the continuous denoiser, we employ rectified flow and choose a FLUX-like architectures, but with much fewer DiT blocks, *i.e.,* 2 double block and 4 single blocks.
* The rectified flow model is built upon the pre-trained latent space of VAE.



## 🔧 Prepare

### Installation

1. Clone this repository and navigate to Continuous folder

   ```shell
   git clone git@github.com:mashijie1028/GenHancer.git
   cd Continuous
   ```

2. Install packages:

   ```shell
   conda create -n continuous python=3.10 -y
   conda activate continuous
   pip install --upgrade pip  # enable PEP 660 support
   pip install -r requirements.txt
   ```

### Dataset

* Download CC3M dataset. Please refer to [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) for more details.
* Specify the dataset path in  `img_dir` of  `train_configs/*.yaml`.

### VAE Model

* Download the VAE model of FLUX.1-dev, namely [FLUX.1-dev/ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors).

* Specify the `AE` environment variable as the local checkpoint path in `train_scripts/*.sh`:

  ```shell
  export AE="YOUR_LOCAL_PATH/FLUX.1-dev/ae.safetensors"
  ```

  



## 🏃 Training

### Stage-1

In Stage-1, we only train the projector and denoiser while freezing the CLIP model.

```shell
# for OpenAICLIP@224
bash train_scripts/scripts_train_OpenAICLIP_224_stage1.sh

# for OpenAICLIP@336
bash train_scripts/scripts_train_OpenAICLIP_336_stage1.sh

# for SigLIP@384
bash train_scripts/scripts_train_SigLIP_384_stage1.sh
```

### Stage-2

In Stage-2, we mainly tune the CLIP model, while we empirically found that whether to freeze the projector and denoiser has little impact on the results.

Please specify `load_dir` as the saved directory of Stage-1 in `train_configs/*.yaml`.

If you tune only CLIP ViT:

```shell
# for OpenAICLIP@224
bash train_scripts/scripts_train_OpenAICLIP_224_stage2_only.sh

# for OpenAICLIP@336
bash train_scripts/scripts_train_OpenAICLIP_336_stage2_only.sh

# for SigLIP@384
bash train_scripts/scripts_train_SigLIP_384_stage2_only.sh
```

If you want to tune all the components:

```shell
# for OpenAICLIP@224
bash train_scripts/scripts_train_OpenAICLIP_224_stage2_all.sh

# for OpenAICLIP@336
bash train_scripts/scripts_train_OpenAICLIP_336_stage2_all.sh

# for SigLIP@384
bash train_scripts/scripts_train_SigLIP_384_stage2_all.sh
```



## 🤗 Acknowledgements

When building the codebase of continuous denosiers, we refer to [x-flux](https://github.com/XLabs-AI/x-flux). Thanks for their wonderful project. Notably, we do NOT use their pre-trained weights.
