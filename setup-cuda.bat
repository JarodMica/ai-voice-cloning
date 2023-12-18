git submodule init
git submodule update --remote

python -m venv venv
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r .\modules\tortoise-tts\requirements.txt
python -m pip install -e .\modules\tortoise-tts\
python -m pip install -r .\modules\dlas\requirements.txt
python -m pip install -e .\modules\dlas\
python -m pip install -r .\requirements.txt
python -m pip install deepspeed-0.8.3+6eca037c-cp39-cp39-win_amd64.whl

# setup BnB
.\setup-cuda-bnb.bat

del *.sh

pause
deactivate
