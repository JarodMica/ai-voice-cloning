#!/bin/bash
git submodule init
git submodule update --remote
docker build -t ai-voice-cloning .
