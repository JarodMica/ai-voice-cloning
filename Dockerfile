FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=UTC
ARG PYTHON_VERSION=3.11.10
ARG UID=1001
ARG GID=1001

# Repositories
ARG BASE_REPO=https://github.com/labay-adam/ai-voice-cloning
ARG RVC_REPO=https://huggingface.co/Jmica/rvc/resolve/main/rvc_lightweight.zip?download=true
ARG FAIRSEQ_REPO=https://github.com/VarunGumma/fairseq
ARG PYFASTMP3DECODER_REPO=https://github.com/neonbjb/pyfastmp3decoder.git
ARG PIPELINE_REPO=https://github.com/JarodMica/rvc-tts-pipeline.git@lightweight#egg=rvc_tts_pipe
ARG WHISPERX_REPO=https://github.com/m-bain/whisperx.git@v3.2.0

# TZ
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Prereqs
RUN apt-get update
RUN apt-get install -y \
    curl \
    wget \
    git \
    ffmpeg \
    p7zip-full \
    unzip \
    gcc \
    g++ \
    nano
    
# Python Prereqs
RUN apt-get install -y \
	libssl-dev \
	liblzma-dev \
	libsqlite3-dev \
	libctypes-ocaml-dev \
	libffi-dev \
	libncurses-dev \
	libbz2-dev \
	libreadline-dev \
	tk-dev \
	make \
	build-essential \
	zlib1g-dev \
	llvm \
	xz-utils \
	ninja-build
	
# AV Prereqs
RUN apt-get install -y \
	libavformat-dev \
	libavcodec-dev \
	libavdevice-dev \
	libavutil-dev \
	libswscale-dev \
	libswresample-dev

# User
RUN groupadd --gid $GID user
RUN useradd --no-log-init --create-home --shell /bin/bash --uid $UID --gid $GID user
USER user
ENV HOME=/home/user
WORKDIR $HOME
RUN mkdir $HOME/.cache $HOME/.config && chmod -R 777 $HOME

# Python
RUN curl https://pyenv.run/ | bash
ENV PYENV=$HOME/.pyenv/bin
RUN $PYENV/pyenv install $PYTHON_VERSION
ENV PYTHON3_BIN=$HOME/.pyenv/versions/$PYTHON_VERSION/bin/python3
USER root
RUN ln -sf $PYTHON3_BIN /usr/bin/python3
USER user

# Prereqs
RUN python3 -m pip install --upgrade pip && \
	python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Base path
RUN git clone $BASE_REPO
WORKDIR $HOME/ai-voice-cloning

# Built in modules
RUN git submodule init &&\
	git submodule update --remote
RUN python3 -m pip install -r ./modules/tortoise-tts/requirements.txt && \
	python3 -m pip install -e ./modules/tortoise-tts/ && \
	python3 -m pip install -r ./modules/dlas/requirements.txt && \
	python3 -m pip install -e ./modules/dlas/

# Stage other modules
RUN curl -L $RVC_REPO -o rvc.zip && \
	python3 -m zipfile -e rvc.zip ./
RUN git clone $FAIRSEQ_REPO && \
	python3 -m pip wheel ./fairseq -w ./fairseq/wheels
RUN git clone --recurse-submodules $PYFASTMP3DECODER_REPO && \
	python3 -m pip wheel ./pyfastmp3decoder -w ./pyfastmp3decoder/wheels

# Install dependencies
RUN python3 -m pip install -r ./rvc/requirements.txt && \
	python3 -m pip install ./fairseq/wheels/fairseq-*.whl && \
	python3 -m pip install git+$PIPELINE_REPO && \
	python3 -m pip install deepspeed && \
	python3 -m pip install ./pyfastmp3decoder/wheels/pyfastmp3decoder-*.whl && \
	python3 -m pip install git+$WHISPERX_REPO && \
	python3 -m pip install -r requirements.txt

ENV IN_DOCKER=true
ENV PYTHON_VERSION=$PYTHON_VERSION
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["./start.sh"]
