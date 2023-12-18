#!/bin/bash
git pull
git submodule update --remote

source ./venv/bin/activate
if python -m pip show whispercpp &>/dev/null; then python -m pip install -U git+https://git.ecker.tech/lightmare/whispercpp.py; fi
deactivate