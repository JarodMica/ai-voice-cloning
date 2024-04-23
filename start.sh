#!/bin/bash
ulimit -Sn `ulimit -Hn` # ROCm is a bitch
while [ true ]; do
    python3 ./src/main.py "$@"
    echo "Press Cntrl-C to quit or application will restart... (5s)"
    sleep 5
done
