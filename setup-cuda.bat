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
python -m pip install deepspeed-0.8.3+6eca037c-cp39-cp39-win_amd64.whl

setlocal enabledelayedexpansion

set file_name=rvc.zip
set download_rvc=https://huggingface.co/Jmica/rvc/resolve/main/rvc_lightweight.zip?download=true
set extracted_folder=rvc

if exist "%extracted_folder%" (
    echo The folder %extracted_folder% already exists.
    choice /C YN /M "Do you want to delete it and re-extract (Y/N)?"
    if errorlevel 2 goto SkipDeletion
    if errorlevel 1 (
        echo Deleting %extracted_folder%...
        rmdir /S /Q "%extracted_folder%"
    )
)

:SkipDeletion

if not exist "%file_name%" (
    echo Downloading %file_name%...
    curl -L %download_rvc% -o %file_name%
) else (
    echo File %file_name% already exists, skipping download.
)

echo Extracting %file_name%...
tar -xf %file_name%

echo RVC has finished downloading and Extracting

rem setup rvc stuff
python -m pip install -r .\rvc\requirements.txt

set download_fairseq=https://huggingface.co/Jmica/rvc/resolve/main/fairseq-0.12.2-cp39-cp39-win_amd64.whl?download=true
set file_name=fairseq-0.12.2-cp39-cp39-win_amd64.whl

if not exist "%file_name%" (
    echo Downloading %file_name%...
    curl -L -O "%download_fairseq%"
    if errorlevel 1 (
        echo Download failed. Please check your internet connection or the URL and try again.
        exit /b 1
    )
) else (
    echo File %file_name% already exists, skipping download.
)

python -m pip install .\fairseq-0.12.2-cp39-cp39-win_amd64.whl
python -m pip install git+https://github.com/JarodMica/rvc-tts-pipeline.git@lightweight#egg=rvc_tts_pipe

rem Need this last because the other requirements break compatible packages
python -m pip install -r .\requirements.txt

# setup BnB
.\setup-cuda-bnb.bat

del *.sh

pause
deactivate
