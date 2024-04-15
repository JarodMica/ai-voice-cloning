#!/bin/bash

function main() {
    if [ ! -f modules/tortoise-tts/README.md ]; then
        git submodule init
        git submodule update --remote
    fi
    docker build \
        --build-arg UID=$(id -u) \
        --build-arg GID=$(id -g) \
        -t ai-voice-cloning \
        .
}

main
