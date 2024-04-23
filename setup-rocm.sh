#!/bin/bash
# get local dependencies
git submodule init
git submodule update --remote
# setup venv
python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install --upgrade pip # just to be safe
# ROCM
pip3 install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.2 # 5.4.2 doesn't work for me desu
# install requirements
python3 -m pip install -r ./modules/tortoise-tts/requirements.txt # install TorToiSe requirements
python3 -m pip install -e ./modules/tortoise-tts/ # install TorToiSe
python3 -m pip install -r ./modules/dlas/requirements.txt # instal DLAS requirements
python3 -m pip install -e ./modules/dlas/ # install DLAS
python3 -m pip install -r ./requirements.txt # install local requirements

rm *.bat

./setup-rocm-bnb.sh

deactivate