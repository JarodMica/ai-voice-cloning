#!/bin/bash
# get local dependencies
git submodule init
git submodule update --remote
# setup venv
python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install --upgrade pip # just to be safe
# CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install requirements
python3 -m pip install -r ./modules/tortoise-tts/requirements.txt # install TorToiSe requirements
python3 -m pip install -e ./modules/tortoise-tts/ # install TorToiSe
python3 -m pip install -r ./modules/dlas/requirements.txt # instal DLAS requirements, last, because whisperx will break a dependency here
python3 -m pip install -e ./modules/dlas/ # install DLAS
python3 -m pip install -r ./requirements.txt # install local requirements

rm *.bat

deactivate