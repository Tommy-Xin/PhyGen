# GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers (ICCV 2025)

**[Shijie Ma<sup>1,2</sup>](https://mashijie1028.github.io/), 
[Yuying Ge<sup>1,&#9993;</sup>](https://geyuying.github.io/), 
[Teng Wang<sup>1</sup>](http://ttengwang.com/), 
[Yuxin Guo<sup>1,2</sup>](https://scholar.google.com/citations?user=x_0spxgAAAAJ&hl=en), 
[Yixiao Ge<sup>1</sup>](https://geyixiao.com/), 
[Ying Shan<sup>1</sup>](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)**
<br>
<sup>1</sup>ARC Lab, Tencent PCG, 
<sup>2</sup>Institute of Automation, CAS
<br>

<a href="https://iccv.thecvf.com/Conferences/2025"><img src="https://img.shields.io/badge/-ICCV%202025-68488b"></a>
<a href='https://arxiv.org/abs/2503.19480'><img src='https://img.shields.io/badge/ArXiv-2503.19480-red'></a> 
<a href='https://mashijie1028.github.io/GenHancer/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://huggingface.co/msj9817/GenHancer/'><img src='https://img.shields.io/badge/Model-Huggingface-yellow'></a>



## ⚡ TL;DR

How do generative models effectively help discriminative models?

We present in-depth explorations and propose a novel two-stage post-training strategy to enhance CLIP ViT's visual representations.

Our method is applicable to both continuous and discrete denoiser without the requirement for pre-trained weights.



## 📅 News
* [2025-06-26] GenHancer is accepted to ICCV 2025! 🎉🎉🎉
* [2025-03-27] Training codes with continuous denoisers are released! 🔥🔥🔥
* [2025-03-26] arXiv paper is made publicly available.
* [2025-03-24] Release evaluation codes. 🔥
* [2025-03-24] Release models weights on [Huggingface🤗](https://huggingface.co/msj9817/GenHancer/). 🔥🔥🔥
* [2025-03-24] Release the [project page](https://mashijie1028.github.io/GenHancer/) of this repo.



## 🔜 TODOs

- [x] Release training codes of continuous denoisers.
- [ ] Release training codes of discrete denoisers.



## 🔎 Introduction

Recent works demonstrate the feasibility of enhancing visual representations with generative models, where generative models take visual tokens as conditions and perform reconstruction. However, the underlying principle remains underexplored.

We empirically reveal that perfect generation (reconstruction) does not always yield desirable visual representations, as shown below:

![teaser](assets/teaser.jpg)

In this work, we delve into three aspects to explore the critical factors: (1) conditioning mechanisms, (2) denoising configurations and (3) generation paradigms.

We propose a two-stage post-training method to enhance CLIP ViT's fine-grained visual representations, which is efficient (with only lightweight denoisers) and versatile (applicable to both continuous and discrete denoisers). The pipeline of our method is illustrated below:

![teaser](assets/method.jpg)




> [!Important]
>
> We empirically found that, for visual representations, a *visually* perfect generative model is not optimal and not necessary. 
>
> Our method only employs lightweight generative models and does NOT require any pre-trained weights, which is efficient and  could avoid potential privacy and copyright issues.





##  ⭐ Released Weights

We release the enhanced CLIP weights  on [Huggingface🤗](https://huggingface.co/msj9817/GenHancer/).

| CLIP Backbone           | MMVP-VLM (Original) | MMVP-VLM (Ours) |                             Link                             |
| :---------------------- | :-----------------: | :-------------: | :----------------------------------------------------------: |
| OpenAICLIP ViT-L-14@224 |        19.3         |      31.9       | [🤗](https://huggingface.co/msj9817/GenHancer/tree/main/OpenAICLIP/clip-vit-large-patch14) |
| OpenAICLIP ViT-L-14@336 |        20.0         |      29.6       | [🤗](https://huggingface.co/msj9817/GenHancer/tree/main/OpenAICLIP/clip-vit-large-patch14-336) |
| MetaCLIP ViT-L-14@224   |        23.7         |      31.9       | [🤗](https://huggingface.co/msj9817/GenHancer/tree/main/MetaCLIP/metaclip-l14-fullcc2.5b) |
| MetaCLIP ViT-H-14@224   |        25.2         |      37.0       | [🤗](https://huggingface.co/msj9817/GenHancer/tree/main/MetaCLIP/metaclip-h14-fullcc2.5b) |
| SigLIP ViT-SO-14@224    |        37.8         |      42.2       | [🤗](https://huggingface.co/msj9817/GenHancer/tree/main/SigLIP/siglip-so400m-patch14-224) |
| SigLIP ViT-SO-14@384    |        37.0         |      40.0       | [🤗](https://huggingface.co/msj9817/GenHancer/tree/main/SigLIP/siglip-so400m-patch14-384) |



## 🏃 Training

Please come into the corresponding directories for more details.

For the continuous denoiser, navigate into [Continuous](https://github.com/mashijie1028/GenHancer/tree/main/Continuous).

For the discrete denoiser, navigate into [Discrete](https://github.com/mashijie1028/GenHancer/tree/main/Discrete).



## 📏 Evaluation

Please first download the benchmark [MMVP-VLM](https://huggingface.co/datasets/MMVP/MMVP_VLM).

We provide evaluation scripts of six CLIP backbones. The example of OpenAICLIP@224 is as follows: 

```shell
python evaluation/evaluate_mmvp_OpenAICLIP_224.py --benchmark_dir 'YOUR_MMVP_VLM_PATH' --vision_tower_name 'YOUR_VISION_TOWER'
```

> [!note]
>
> Please specify `--vision_tower_name` as your trained CLIP model, which is conventionally saved via `save_pretrained()`.
>
> If you want to evaluation the official CLIP model like OpenAICLIP@224, you could specify `--vision_tower_name` as the official `hf_repo_id`, *e.g.,* `openai/clip-vit-large-patch14`.



## 🤗 Acknowledgements

When building the codebase of continuous denosiers, we refer to [x-flux](https://github.com/XLabs-AI/x-flux). Thanks for their wonderful project. Notably, we do NOT use their pre-trained weights.



## 📜 License

This repository is under the [Apache 2 License](https://github.com/mashijie1028/Gen4Rep/blob/main/LICENSE).



## 📚 BibTeX

```
@article{ma2025genhancer,
	title={GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers},
	author={Ma, Shijie and Ge, Yuying and Wang, Teng and Guo, Yuxin and Ge, Yixiao and Shan, Ying},
	journal={arXiv preprint arXiv:2503.19480},
	year={2025}
}
```



## 📧 Contact

If you have further questions, feel free to contact me: mashijie9817@gmail.com

Discussions and potential collaborations are also welcome.
