#!/bin/bash

docker run \
    -ti \
    --rm \
    --gpus all \
    --name ai-voice-cloning \
    -v "${PWD}/models:/home/user/ai-voice-cloning/models" \
    -v "${PWD}/training:/home/user/ai-voice-cloning/training" \
    -v "${PWD}/voices:/home/user/ai-voice-cloning/voices" \
    -v "${PWD}/bin:/home/user/ai-voice-cloning/bin" \
    -v "${PWD}/config:/home/user/ai-voice-cloning/config" \
    --user "$(id -u):$(id -g)" \
    -p "7860:7860" \
    ai-voice-cloning $@

# For dev:
#     -v "${PWD}/src:/home/user/ai-voice-cloning/src" \
#     -v "${PWD}/modules/tortoise_dataset_tools/dataset_whisper_tools:/home/user/ai-voice-cloning/modules/tortoise_dataset_tools/dataset_whisper_tools" \
#     -v "${PWD}/modules/dlas/dlas:/home/user/ai-voice-cloning/modules/dlas/dlas" \
#     -v "/home/user/ai-voice-cloning/src/__pycache__" \

# For testing:
#    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
