@echo off
setlocal enabledelayedexpansion

:: Check if Python 3.11 is installed
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 is not installed. Please install it and try again.
    pause
    exit /b 1
)

:: Initialize and update git submodules
git submodule init
git submodule update

:: Set up virtual environment with Python 3.11
py -3.11 -m venv venv
call .\venv\Scripts\activate.bat

:: Upgrade pip and install required packages
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r .\modules\tortoise-tts\requirements.txt
python -m pip install -e .\modules\tortoise-tts\
python -m pip install -r .\modules\dlas\requirements.txt
python -m pip install -e .\modules\dlas\

:: Download and extract RVC if not already done
set file_name=rvc.zip
set download_rvc=https://huggingface.co/Jmica/rvc/resolve/main/rvc_lightweight.zip?download=true
set extracted_folder=rvc

:: Delete previous rvc.zip if it exists
if exist "%file_name%" (
  echo Deleting previous %file_name%...
  del "%file_name%"
)

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
echo RVC has finished downloading and Extracting.

:: Install RVC requirements
python -m pip install -r .\rvc\requirements.txt

:: Download and install Fairseq if not already done
set download_fairseq=https://huggingface.co/Jmica/rvc/resolve/main/fairseq-0.12.4-cp311-cp311-win_amd64.whl?download=true
set file_name=fairseq-0.12.4-cp311-cp311-win_amd64.whl

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

set download_deepspeed=https://huggingface.co/Jmica/rvc/resolve/main/deepspeed-0.14.0-cp311-cp311-win_amd64.whl?download=true
set fileds_name=deepspeed-0.14.0-cp311-cp311-win_amd64.whl

if not exist "%fileds_name%" (
    echo Downloading %fileds_name%...
    curl -L -O "%download_deepspeed%"
    if errorlevel 1 (
        echo Download failed. Please check your internet connection or the URL and try again.
        exit /b 1
    )
) else (
    echo File %fileds_name% already exists, skipping download.
)

set download_pyfastmp3decoder=https://huggingface.co/Jmica/rvc/resolve/main/pyfastmp3decoder-0.0.1-cp311-cp311-win_amd64.whl?download=true
set filepyfast_name=pyfastmp3decoder-0.0.1-cp311-cp311-win_amd64.whl

if not exist "%filepyfast_name%" (
    echo Downloading %filepyfast_name%...
    curl -L -O "%download_pyfastmp3decoder%"
    if errorlevel 1 (
        echo Download failed. Please check your internet connection or the URL and try again.
        exit /b 1
    )
) else (
    echo File %filepyfast_name% already exists, skipping download.
)

:: Install Fairseq, Deepspeed, pyfast, and RVC TTS Pipeline
python -m pip install .\fairseq-0.12.4-cp311-cp311-win_amd64.whl
python -m pip install git+https://github.com/JarodMica/rvc-tts-pipeline.git@lightweight#egg=rvc_tts_pipe
python -m pip install deepspeed-0.14.0-cp311-cp311-win_amd64.whl
python -m pip install pyfastmp3decoder-0.0.1-cp311-cp311-win_amd64.whl

:: Install whisperx
python -m pip install git+https://github.com/m-bain/whisperx.git

:: Install other requirements (this is done last due to potential package conflicts)
python -m pip install -r requirements.txt

:: Download and install ffmpeg
call download_ffmpeg.bat

:: Setup BnB
call setup-cuda-bnb.bat

.\start.bat
:: Clean up
del *.sh

pause
deactivate
