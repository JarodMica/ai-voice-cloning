#!/bin/bash
CMD="python3 ./src/train.py --yaml $1"
# ipc host is one way to increase the shared memory for the container
# more info here https://github.com/pytorch/pytorch#docker-image
CPATH="/home/user/ai-voice-cloning"
docker run --rm --gpus all \
    --mount "type=bind,src=$PWD/models,dst=$CPATH/models" \
    --mount "type=bind,src=$PWD/training,dst=$CPATH/training" \
    --mount "type=bind,src=$PWD/voices,dst=$CPATH/voices" \
    --mount "type=bind,src=$PWD/bin,dst=$CPATH/bin" \
    --mount "type=bind,src=$PWD/src,dst=$CPATH/src" \
    --workdir $CPATH \
    --ipc host \
    --user "$(id -u):$(id -g)" \
    -it ai-voice-cloning $CMD
