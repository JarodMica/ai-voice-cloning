FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=UTC
ARG MINICONDA_VERSION=23.1.0-1
ARG PYTHON_VERSION=3.11
ARG UID=1000
ARG GID=1000

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
    gcc \
    g++ \
    vim

# User
RUN groupadd --gid $GID user
RUN useradd --no-log-init --create-home --shell /bin/bash --uid $UID --gid $GID user
USER user
ENV HOME=/home/user
WORKDIR $HOME
RUN mkdir $HOME/.cache $HOME/.config && chmod -R 777 $HOME

# Python
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_$MINICONDA_VERSION-Linux-x86_64.sh
RUN chmod +x Miniconda3-py39_$MINICONDA_VERSION-Linux-x86_64.sh
RUN ./Miniconda3-py39_$MINICONDA_VERSION-Linux-x86_64.sh -b -p /home/user/miniconda
ENV PATH="$HOME/miniconda/bin:$PATH"
RUN conda init
RUN conda install python=$PYTHON_VERSION
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Base path
RUN mkdir $HOME/ai-voice-cloning
WORKDIR $HOME/ai-voice-cloning

# Built in modules
COPY --chown=user:user modules modules
RUN python3 -m pip install -r ./modules/tortoise-tts/requirements.txt
RUN python3 -m pip install -e ./modules/tortoise-tts/
RUN python3 -m pip install -r ./modules/dlas/requirements.txt
RUN python3 -m pip install -e ./modules/dlas/

# RVC
RUN \
    curl -L -o /tmp/rvc.zip https://huggingface.co/Jmica/rvc/resolve/main/rvc_lightweight.zip?download=true &&\
    7z x /tmp/rvc.zip &&\
    rm -f /tmp/rvc.zip
USER root
RUN \
    chown user:user rvc -R &&\
    chmod -R u+rwX,go+rX,go-w rvc
USER user
RUN python3 -m pip install -r ./rvc/requirements.txt

# Fairseq
# Using patched version for Python 3.11 due to https://github.com/facebookresearch/fairseq/issues/5012
RUN python3 -m pip install git+https://github.com/liyaodev/fairseq

# RVC Pipeline
RUN python3 -m pip install git+https://github.com/JarodMica/rvc-tts-pipeline.git@lightweight#egg=rvc_tts_pipe

# Deepspeed
RUN python3 -m pip install deepspeed

# PyFastMP3Decoder
RUN python3 -m pip install cython
RUN git clone https://github.com/neonbjb/pyfastmp3decoder.git
RUN \
    cd pyfastmp3decoder &&\
    git submodule update --init --recursive &&\
    python setup.py install &&\
    cd ..

# WhisperX
RUN python3 -m pip install git+https://github.com/m-bain/whisperx.git

# Main requirements
ADD requirements.txt requirements.txt
RUN python3 -m pip install -r ./requirements.txt

# The app
ADD --chown=user:user . $HOME/ai-voice-cloning

ENV IN_DOCKER=true

CMD ["./start.sh"]
