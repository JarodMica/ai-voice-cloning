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
    --net host \
    ai-voice-cloning

# For dev:
#     -v "${PWD}/src:/home/user/ai-voice-cloning/src" \
#     -v "/home/user/ai-voice-cloning/src/__pycache__" \
