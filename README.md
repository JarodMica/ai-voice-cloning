# AI Voice Cloning

> **Note** I do not plan on actively working on improvements/enhancements for this project, this is mainly meant to keep the repo in a working state in the case the original git.ecker goes down or necessary package changes need to be made.

That being said, some enhancements added compared to the original repo:

:heavy_check_mark: Hifigan added, allowing for faster inference at the cost of quality.  

:heavy_check_mark: whisper-v3 added as a chooseable option for whisperx

:heavy_check_mark: Output conversion using RVC 

This is a fork of the repo originally located here: https://git.ecker.tech/mrq/ai-voice-cloning.  All of the work that was put into it to incoporate training with DLAS and inference with Tortoise belong to mrq, the author of the original ai-voice-cloning repo.  

## Setup (Windows + Nvidia)
This repo only works on **Windows with NVIDIA GPUs**.  I don't have any plans on making it compatible with other systems, but it shouldn't be too difficult to port if you have experience in coding or are an expert level ChatGPT user.  If you do successfully do this and want to share, pull requests are always welcome!
> **Tips for developers:** setup-cuda.bat should have everything that you need for the packages to be installed.  All of the different requirements files make it quite a mess in the script, but each repo has their requirements installed, and then at the end, the requirements.txt in the root is needed to change the version *back* to compatible versions for this repo.

### Package Installation (Recommended)
Install 7zip on your computer: https://www.7-zip.org/
    - If you run into any extraction issues, most likely it's due to your 7zip being out-of-date OR you're using a different extractor.

1. Head over to the releases tab and download the latest package on Hugging Face:
2. Extract the 7zip archive.
3. Open up ai-voice-cloning and then run ```start.bat```

### Manual Installation
If you are installing this manually, you will need:
- Python 3.9: https://www.python.org/downloads/release/python-3913/
- Git: https://www.git-scm.com/downloads

1. Clone the repository
```
git clone https://github.com/JarodMica/ai-voice-cloning.git
```
2. Run the ```setup-cuda.bat``` file and it will start running through all of the python packages needed
    - If you don't have python 3.9, it won't work and you'll need to go download it
3. After it finishes, run ```start.bat``` and this will start downloading most of the models you'll need.
    - Some models are downloaded when you first use them.  You'll incur additional downloads during generation and when training (for whisper).  However, once they are finished, you won't ever have to download them again as long as you don't delete them.  They are located in the ```models``` folder of the root.
4. **(Optional)** You can opt to install whisperx for training by running ```setup-whipserx.bat```
    - Check out the whisperx github page for more details, but it's much faster for longer audio files.  If you're processing one-by-one with an already split dataset, it doesn't improve speeds that much.

## Instructions
Checkout the YouTube video: insert_link_here_later

## Updating Your Installation
Below are how you can update the package for the latest updates

### Package
>**NOTE:** If there are major feature change, check the latest release to see if ```update_package.bat``` will work.  If NOT, you will need to re-download and re-extract the package from Hugging Face.
1. Run the update_package.bat file
    - It will clone the repo and will copy the src folder from the repo to the package.

### Manual Installation
You should be able to navigate into the folder and then pull the repo to update it.
```
cd ai-voice-cloning
git pull
```
If there are large features added, you may need to delete the venv and the re-run the setup-cuda script to make sure there are no package issues

## Documentation

### Troubleshooting Manual Installation
The terminal is your friend.  Any errors or issues will pop-up in the terminal when you go to try and run, and then you can start debugging from there.
- If somewhere in the process, torch gets messed up, you may have to reinstall it.  You will have to uninstall it, then reinstall it like the following.  Make sure to type (Y) to confirm deletion.

```
.\venv\Scripts\activate.bat
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Other documentation tbd

## Bug Reporting

If you run into any problems, please open up a new issue on the issues tab.
