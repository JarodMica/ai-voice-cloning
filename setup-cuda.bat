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

# setup BnB
.\setup-cuda-bnb.bat

del *.sh

pause
deactivate
