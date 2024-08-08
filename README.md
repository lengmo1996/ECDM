:new: [2024-07-30] upload base resource code
# ECDM <!-- omit in toc -->
This is the official repository of our work: Data Generation Scheme for Thermal Modality with Edge-Guided Adversarial Conditional Diffusion Model (ACM MM'24)

[Guoqing Zhu](https://github.com/lengmo1996), Honghu Pan, [Qaing Wang](https://blackjack2015.github.io/), Chao Tian, Chao Yang, [Zhenyu He](https://www.hezhenyu.cn/)

[ACM MM'24](https://openreview.net/forum?id=GSmdnRqbpD) | arXiv | [GitHub](https://github.com/lengmo1996/ECDM) | Project Page




## TODO list <!-- omit in toc -->
:heavy_check_mark: upload base resource code

:x: update weights

:heavy_check_mark: update related datasets

:heavy_check_mark: update datasets processing scripts

:heavy_check_mark: update evaluation scripts

:x: update generated thermal images from different methods

:x: support more sampler
  
## Update <!-- omit in toc -->
- [2024-07-30] upload base resource code

## Contents <!-- omit in toc -->
- [User guide](#user-guide)
  - [1. Environment configuration](#1-environment-configuration)
    - [1.1 Docker (Recommended)](#11-docker-recommended)
    - [1.2 Conda (Recommended)](#12-conda-recommended)
    - [1.3 Pip (Python \>= 3.10)](#13-pip-python--310)
  - [2. Dataset Preparation](#2-dataset-preparation)
    - [2.1 Datasets](#21-datasets)
    - [2.2 Generating edge images](#22-generating-edge-images)
  - [3. Training](#3-training)
    - [3.1 Training first stage model](#31-training-first-stage-model)
    - [3.2 Training second stage model](#32-training-second-stage-model)
    - [3.3 Resuming training process](#33-resuming-training-process)
  - [4. Evaluation](#4-evaluation)
    - [4.1 Generating themal images](#41-generating-themal-images)
    - [4.2 Evaluation](#42-evaluation)
  - [5. Computing model Flops](#5-computing-model-flops)
  - [6. Download links](#6-download-links)
    - [6.1 Processed datasets](#61-processed-datasets)
    - [6.2 Model weights](#62-model-weights)
    - [6.3 Generated thermal images from other methods](#63-generated-thermal-images-from-other-methods)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)


## User guide

### 1. Environment configuration
We offer guides on how to install dependencies via docker and conda.

First, clone the reposity:
```bash
git clone https://github.com/lengmo1996/ECDM.git
cd ECDM
```
Then, use one of the below methods to configure the environment.
#### 1.1 Docker (Recommended)
Building image from Dockerfile
```bash
# build image from Dockerfile
docker build -t ecdm:1.0 .
# or from the mirrors
docker build -t ecdm:1.0 -f Dockerfile_mirror
```
or pulling from Docker Hub
```bash
docker pull lengmo1996/ecdm:1.0
```
Then creating container from docker image.
```bash
docker run -it --shm-size 100g --gpus all -v /path_to_dataset:/path_to_dataset -v /path_to_log:/path_to_log -v /path_to_ECDM:/path_to_ECDM --name ECDM ecdm:1.0 /bin/bash
```

#### 1.2 Conda (Recommended)
```bash
conda env create -f conda.yaml
conda activate ecdm
```
#### 1.3 Pip (Python >= 3.10)

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 
pip install -r requirements.txt
```


### 2. Dataset Preparation
#### 2.1 Datasets
We use LLVIP and PRW dataset in our experiments. You can download LLVIP dataset from [here](https://bupt-ai-cz.github.io/LLVIP/) and PRW dataset from [here](https://github.com/liangzheng06/PRW-baseline).

#### 2.2 Generating edge images
We use [scripts/generate_edge_img.py](scripts/generate_edge_img.py) to generate edge images. You should modify the ==lines 19-20== to specify your custom dataset path. Then, execute the following command to generate the edge images::
```bash
python scripts/generate_edge_img.py
```
This method applies a high-pass filter to the original images. Edge images can also be obtained by other edge detection methods, such as [pidinet](https://github.com/hellozhuo/pidinet) or [teed](https://github.com/xavysp/TEED).


### 3. Training
#### 3.1 Training first stage model
Please modify the ==lines 66, 73, 81== in [configs/ecdm_first_stage.yaml](configs/ecdm_first_stage.yaml) to specify your custom paths. Then, run the following command to train the first stage model:
```bash
python main.py fit -c configs/base_config.yaml -c configs/ecdm_first_stage.yaml --trainer.devices 0,1,2,3 
```
#### 3.2 Training second stage model
Please modify the ==lines 74, 81, 89== in [configs/ecdm_second_stage.yaml](configs/ecdm_second_stage.yaml) to your custom paths. Additionally, modify ==line 16== to point to the path of the first stage model weight. Then, run the following command to train the second stage model:
```bash 
python main.py fit -c configs/base_config.yaml -c configs/ecdm_second_stage.yaml --trainer.devices 0,1,2,3 
```
If you want to generalize the model to the PRW dataset, please use the configuration of [configs/ecdm_second_stage_with_PRW.yaml](configs/ecdm_second_stage_with_PRW.yaml).

#### 3.3 Resuming training process
If you want to resume the training process, please use the argument '--ckpt_path'. For example:
```bash
python main.py fit -c configs/base_config.yaml -c configs/ecdm_second_stage.yaml --trainer.devices 0,1,2,3 --ckpt_path 'logs/checkpoints/last.ckpt'
```

### 4. Evaluation
#### 4.1 Generating themal images
For evaluation, we need to run the following command to generate thermal images:
```bash
python main.py test -c configs/base_config.yaml -c configs/ecdm_second_stage.yaml --trainer.devices 0,1,2,3 --ckpt_path 'logs/checkpoints/last.ckpt'
```

#### 4.2 Evaluation
Please modify the ==lines 196-198== in [scripts/metrics.py](scripts/metrics.py) to your custom paths. Then, run the following command to evaluate the second stage model:
```bash 
python scripts/metrics.py
```
Note: If version of scikit-image >= 0.16, you may trigger the following error
```bash
cannot import name 'compare_ssim' from 'skimage.measure'. 
```
To fix this bug, you may need to modify the lines ==23-25== in /opt/conda/lib/python3.10/site-packages/lpips/\_\_init\_\_.py.
```bash
def dssim(p0, p1, range=255.):
    from skimage.metrics import structural_similarity
    return (1 - structural_similarity(p0, p1, data_range=range, channel_axis=2)) / 2.
```
More information can be found in [GitHub issue](https://github.com/williamfzc/stagesepx/issues/150) and [release note of scikit-image 0.16.2 (2019-10-22)](https://scikit-image.org/docs/stable/release_notes/release_0.16.html#scikit-image-0-16-1-2019-10-11).
### 5. Computing model Flops
You can obtain the number of parameters and FLOPs of the sampling process using the script located at [scripts/compute_flops_macs_params.py](scripts/compute_flops_macs_params.py)

### 6. Download links
#### 6.1 Processed datasets
Baidu Drive: [LLVIP](https://pan.baidu.com/s/1Py5IJWVRAGDAYZqzYxaMlA?pwd=uh21), [PRW](https://pan.baidu.com/s/1jRaL7_euDliu9XV0YDNR6A?pwd=45hb)
#### 6.2 Model weights

#### 6.3 Generated thermal images from other methods




## Acknowledgments
This implementation is based on / inspired by: [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [improved-diffusion](https://github.com/openai/improved-diffusion) and [latent-diffusion](https://github.com/CompVis/latent-diffusion), 
## Citation
If you find our work helpful for your research, please consider citing the following BibTeX entry.