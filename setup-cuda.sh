#!/bin/bash

# Check if Python 3.11 is installed
python_bin='python3.11'

if ! command -v $python_bin &> /dev/null; then
    echo "Python 3.11 is not installed. Please install it using the following command:"
    echo "sudo apt install -y python3.11"
    echo "After installing Python 3.11, please run the script again."
    exit 1
fi

# Check if python3.11-venv is installed
if ! $python_bin -m venv --help &> /dev/null; then
    echo "The python3.11-venv package is not installed. Please install it using the following command:"
    echo "sudo apt install -y python3.11-venv"
    echo "After installing the python3.11-venv package, please run the script again."
    exit 1
fi

# Set up virtual environment with Python 3.11
$python_bin -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Please check the error message above and try again."
    exit 1
fi

# Initialize and update git submodules
git submodule init
git submodule update --remote

# Upgrade pip and install required packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r ./modules/tortoise-tts/requirements.txt
pip install -e ./modules/tortoise-tts/
pip install -r ./modules/dlas/requirements.txt
pip install -e ./modules/dlas/

# Download and extract RVC if not already done
file_name='rvc.zip'
download_rvc='https://huggingface.co/Jmica/rvc/resolve/main/rvc_lightweight.zip?download=true'
extracted_folder='rvc'

# Delete previous rvc.zip if it exists
if [ -f $file_name ]; then
  echo "Deleting previous ${file_name}..."
  rm -f $file_name
fi

if [ -d $extracted_folder ]; then
    echo "The folder ${extracted_folder} already exists."
    read -p "Do you want to delete it and re-extract? [y/N] " choice
    if [[ $choice == [Yy]* ]]; then
        echo "Deleting ${extracted_folder}..."
        rm -rf $extracted_folder
    fi
fi

if ! [ -f $file_name ]; then
    echo "Downloading ${file_name}..."
    curl -L $download_rvc -o $file_name
else
    echo "File ${file_name} already exists, skipping download."
fi

echo "Extracting ${file_name}..."
$python_bin -m zipfile -e $file_name ./
echo "RVC has finished downloading and Extracting."

# Install RVC requirements
pip install -r ./rvc/requirements.txt

# Prepare fairseq
fairseq_repo='https://github.com/VarunGumma/fairseq'
fairseq_folder='fairseq'

if [ -d $fairseq_folder ]; then
    git -C $fairseq_folder pull
else
    git clone $fairseq_repo
fi

if [ -d $fairseq_folder/wheels ]; then
    rm -rf $fairseq_folder/wheels
fi

# Prepare pyfastmp3decoder
pyfastmp3decoder_repo='https://github.com/neonbjb/pyfastmp3decoder.git'
pyfastmp3decoder_folder='pyfastmp3decoder'

if [ -d $pyfastmp3decoder_folder ]; then
    git -C $pyfastmp3decoder_folder pull
else
    git clone --recurse-submodules $pyfastmp3decoder_repo
fi

if [ -d $pyfastmp3decoder_folder/wheels ]; then
    rm -rf $pyfastmp3decoder_folder/wheels
fi

# Install Fairseq, Deepspeed, pyfast, and RVC TTS Pipeline
pip wheel ./$fairseq_folder/ -w ./$fairseq_folder/wheels/
pip install ./$fairseq_folder/wheels/fairseq-*.whl
pip install git+https://github.com/JarodMica/rvc-tts-pipeline.git@lightweight#egg=rvc_tts_pipe
pip install deepspeed
pip wheel ./$pyfastmp3decoder_folder/ -w ./$pyfastmp3decoder_folder/wheels/
pip install ./$pyfastmp3decoder_folder/wheels/pyfastmp3decoder-*.whl

# Install whisperx
pip install git+https://github.com/m-bain/whisperx.git

# Install other requirements (this is done last due to potential package conflicts)
pip install -r requirements.txt

chmod +x ./start.sh
./start.sh
# Clean up
rm -f *.bat

deactivate
