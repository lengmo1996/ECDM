FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

MAINTAINER lengmo <guoqingzhu1996@gmail.com>

ENV DEBIAN_FRONTEND noninteractive

RUN rm /bin/sh && ln -s /bin/bash /bin/sh \
    && buildDeps='git vim' \
    && apt-get update && apt install -y $buildDeps libgl1-mesa-glx libglib2.0-0 \
    && pip install lightning==2.2.2 \
    && pip install omegaconf==2.1.1 \
    && pip install einops==0.8.0 tqdm \
    && pip install taming-transformers-rom1504 \
    && pip install jsonargparse[signatures]==4.32.0 \
    && pip install tensorboard==2.17.0 \
    && pip install opencv-python==4.10.0.84 \
    && pip install calflops \
    && pip install clean-fid==0.1.35 \
    && pip install lpips \
    && pip install git+https://github.com/openai/CLIP.git \
    && pip install scikit-image \
    && pip install transformers \
    && apt-get autoclean \
    && apt-get clean \
    && rm -rf ~/.cache/pip \
    && rm -rf ~/.cache/pip3 \
