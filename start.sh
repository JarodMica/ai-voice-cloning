#!/bin/bash
ulimit -Sn `ulimit -Hn` # ROCm is a bitch
source ./venv/bin/activate
python3 ./src/main.py "$@"
deactivate
