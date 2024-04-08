#!/bin/bash

# Check if Python 3.9 is installed
python_bin='python3.9'
min_python_version='"3.9.0"'

$python_bin -m pip install --upgrade packaging
$python_bin -c "import platform; from packaging.version import Version; exit(Version(platform.python_version()) < Version(${min_python_version}))"
if [[ $? = 1 ]]; then
    echo "Python >= ${min_python_version} is not installed. Please install it and try again."
    exit 1
fi

# Initialize and update git submodules
git submodule init
git submodule update --remote

# Set up virtual environment with Python 3.9
$python_bin -m venv venv
source ./venv/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r ./modules/tortoise-tts/requirements.txt
pip install -e ./modules/tortoise-tts/
pip install -r ./modules/dlas/requirements.txt
pip install -e ./modules/dlas/

# Download and extract RVC if not already done
file_name='rvc.zip'
download_rvc='https://huggingface.co/Jmica/rvc/resolve/main/rvc_lightweight.zip?download=true'
extracted_folder='rvc'

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
fairseq_repo='https://github.com/pytorch/fairseq'
fairseq_folder='fairseq'

if [ -d $fairseq_folder ]; then
    git -C $fairseq_folder pull
else
    git clone $fairseq_repo
fi

if [ -d $fairseq_folder/wheels ]; then
    rm -rf $fairseq_folder/wheels
fi

# Install Fairseq and RVC TTS Pipeline
pip wheel ./$fairseq_folder/ -w ./$fairseq_folder/wheels/
pip install ./$fairseq_folder/wheels/fairseq-*.whl
pip install git+https://github.com/JarodMica/rvc-tts-pipeline.git@lightweight#egg=rvc_tts_pipe

# Install other requirements (this is done last due to potential package conflicts)
pip install -r requirements.txt

# Clean up
rm -f *.bat

deactivate
