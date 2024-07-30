FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

MAINTAINER lengmo <guoqingzhu1996@gmail.com>

RUN rm /bin/sh && ln -s /bin/bash /bin/sh \
        && pip install lightning==2.2.2 \
        && pip install omegaconf==2.1.1 \
        && pip install einops==0.8.0 tqdm \
        && pip install taming-transformers-rom1504 \
        && pip install jsonargparse[signatures]==4.32.0 \
        && pip install tensorboard==2.17.0 \
        && rm -rf ~/.cache/pip \
        && rm -rf ~/.cache/pip3 \
