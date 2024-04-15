#!/bin/bash
source ./venv/bin/activate
python3 ./src/train.py --yaml "$1"
deactivate
