:new: [2024-07-30] upload base resource code
# ECDM <!-- omit in toc -->
This is the official repository of our work: Data Generation Scheme for Thermal Modality with Edge-Guided Adversarial Conditional Diffusion Model (ACM MM'24)

[Guoqing Zhu](https://github.com/lengmo1996), [Honghu Pan](), [Qaing Wang](https://blackjack2015.github.io/), Chao Tian, Chao Yang, [Zhenyu He](https://www.hezhenyu.cn/)

[ACM MM'24]() | [arXiv]() | [GitHub]() | [Project Page]()




## TODO list <!-- omit in toc -->
:heavy_check_mark: upload base resource code

:x: update weights

:x: update related datasets

:x: update datasets processing scripts

:x: update evaluation scripts

:x: update generated thermal images from different methods
  
## Update <!-- omit in toc -->
- [2024-07-30] upload base resource code

## Contents <!-- omit in toc -->
- [User guide](#user-guide)
  - [1. Environment configuration](#1-environment-configuration)
    - [1.1 Docker (Recommended)](#11-docker-recommended)
    - [1.2 Conda (Recommended)](#12-conda-recommended)
    - [1.3 Pip (Python \>= 3.10)](#13-pip-python--310)
  - [2. Data preparing](#2-data-preparing)
    - [2.1 Datasets](#21-datasets)
    - [2.2 Processing datasets](#22-processing-datasets)
  - [3. Training](#3-training)
    - [3.1 Training first stage model](#31-training-first-stage-model)
    - [3.2 Training second stage model](#32-training-second-stage-model)
  - [4. Evaluation](#4-evaluation)
    - [4.1 Generating themal images](#41-generating-themal-images)
    - [4.2 Evaluation](#42-evaluation)
  - [5. Computing model Flops](#5-computing-model-flops)
  - [6. Pre-trained models](#6-pre-trained-models)
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
```
or Pulling from Docker Hub
```bash
docker pull xxx
```
Then creating container from image.
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


### 2. Data preparing
#### 2.1 Datasets

#### 2.2 Processing datasets

### 3. Training
#### 3.1 Training first stage model

#### 3.2 Training second stage model




### 4. Evaluation
#### 4.1 Generating themal images

#### 4.2 Evaluation

### 5. Computing model Flops


### 6. Pre-trained models



## Acknowledgments



This codebase borrows from xxx

## Citation
