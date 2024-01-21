#!/bin/bash
set -e # Stop on any command erroring

git submodule init
git submodule update --remote
docker build -t ai-voice-cloning .
