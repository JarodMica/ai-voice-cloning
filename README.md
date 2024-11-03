# AI Voice Cloning

> **Note** I do not plan on actively working on improvements/enhancements for this project, this is mainly meant to keep the repo in a working state in the case the original git.ecker goes down or necessary package changes need to be made.

That being said, some enhancements added compared to the original repo:

:heavy_check_mark: Possible to train in other languages

:heavy_check_mark: Hifigan added, allowing for faster inference at the cost of quality.  

:heavy_check_mark: whisper-v3 added as a chooseable option for whisperx

:heavy_check_mark: Output conversion using RVC 

This is a fork of the repo originally located here: https://git.ecker.tech/mrq/ai-voice-cloning.  All of the work that was put into it to incoporate training with DLAS and inference with Tortoise belong to mrq, the author of the original ai-voice-cloning repo.  

## Setup
This repo works on **Windows with NVIDIA GPUs** and **Linux running Docker with NVIDIA GPUs**. 

### Windows Package (Recommended)
1. Optional, but recommended: Install 7zip on your computer: https://www.7-zip.org/
    - If you run into any extraction issues, most likely it's due to your 7zip being out-of-date OR you're using a different extractor.
2. Head over to the releases tab and download the latest package on Hugging Face: https://github.com/JarodMica/ai-voice-cloning/releases/tag/v3.0
3. Extract the 7zip archive.
4. Open up ai-voice-cloning and then run ```start.bat```

#### Alternative Manual Installation

If you are installing this manually, you will need:
- Python 3.11: https://www.python.org/downloads/release/python-311/
- Git: https://www.git-scm.com/downloads

1. Clone the repository
```
git clone https://github.com/JarodMica/ai-voice-cloning.git
```
2. Run the ```setup-cuda.bat``` file and it will start running through all of the python packages needed
    - If you don't have python 3.11, it won't work and you'll need to go download it
3. After it finishes, run ```start.bat``` and this will start downloading most of the models you'll need.
    - Some models are downloaded when you first use them.  You'll incur additional downloads during generation and when training (for whisper).  However, once they are finished, you won't ever have to download them again as long as you don't delete them.  They are located in the ```models``` folder of the root.
4. **(Optional)** You can opt to install whisperx for training by running ```setup-whipserx.bat```
    - Check out the whisperx github page for more details, but it's much faster for longer audio files.  If you're processing one-by-one with an already split dataset, it doesn't improve speeds that much.


### Docker for Linux (or WSL2)

#### Linux Specific Setup
1. Make sure the latest nvidia drivers are installed: `sudo ubuntu-drivers install`
2. Install Docker your preferred way. One way to do it is to follow the official documentation [here](https://docs.docker.com/engine/install/ubuntu/#uninstall-old-versions).
    - Start by uninstalling the old versions
    - Follow the "apt" repository installation method
    - Check that everything is working with the "hello-world" container

3. If, when launching the voice cloning docker, you have an error message saying that the GPU cannot be used, you might have to install [Nvidia Docker Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
    - Install with the "apt" method
    - Run the docker configuration command

        ```sudo nvidia-ctk runtime configure --runtime=docker```
    
    - Restart docker




#### Windows Specific Setup
> Make sure your Nvidia drivers are up to date: https://www.nvidia.com/download/index.aspx
1. Install WSL2 in PowerShell with `wsl --install` and restart
2. Open PowerShell, type and enter ```ubuntu```.  It should now load you into wsl2
3. Remove the original nvidia cache key: `sudo apt-key del 7fa2af80`
4. Download CUDA toolkit keyring: `wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb`
5. Install keyring: `sudo dpkg -i cuda-keyring_1.1-1_all.deb`
6. Update package list: `sudo apt-get update`
7. Install CUDA toolkit: `sudo apt-get -y install cuda-toolkit-12-4`
8. Install Docker Desktop using WSL2 as the backend
9. Restart
10. If you wish to monitor the terminal remotely via SSH, follow [this guide](https://www.hanselman.com/blog/how-to-ssh-into-wsl2-on-windows-10-from-an-external-machine).
11. Open PowerShell, type ```ubuntu```, [then follow below](#building-and-running-in-docker)

#### Building and Running in Docker

1. Open a terminal (or Ubuntu WSL)
2. Clone the repository: `git clone https://github.com/JarodMica/ai-voice-cloning.git && cd ai-voice-cloning`
3. Build the image with `./setup-docker.sh`
4. Start the container with `./start-docker.sh`
5. Visit `http://localhost:7860` or remotely with `http://<ip>:7860`

>If remote server cannot be reached, checkout this [thread](https://github.com/JarodMica/ai-voice-cloning/issues/92)

You might also need to remap your local folders to the Docker folders. To do this, you must open the "start-docker.sh" script, and update some lines. For instance, if you want to find your generated audios easily, create a "results" folder in the root directory, and then in "start-docker.sh" add the line:

```-v "your/custom/path:/home/user/ai-voice-cloning/results"```



## Instructions
Checkout the YouTube video:

Watch First: https://youtu.be/WWhNqJEmF9M?si=RhUZhYersAvSZ4wf

Watch Second (RVC update): https://www.youtube.com/watch?v=7tpWH8_S8es&t=504s

Everything is pretty much the same as before if you've used this repository in the past, however, there is a new option to convert text output using ```rvc```.  Before you can use it, you will need a **trained** RVC .pth file that you get from RVC or online, and then you will need to place it in ```models/rvc_models/```.  Both .index and .pth files can be placed in here and they'll show up correctly in their respective dropdown menus.

To enable rvc: 
1. Check and enable ```Show Experimental Settings``` to reveal more options
2. Check and enable ```Run the outputter audio through RVC```.
You will now have access to parameters you could adjust in RVC for the RVC voice model you're using.

## Updating Your Installation
Below are how you can update the package for the latest updates

### Windows
>**NOTE:** If there are major feature change, check the latest release to see if ```update_package.bat``` will work.  If NOT, you will need to re-download and re-extract the package from Hugging Face.
1. Run the `update_package.bat `file
    - It will clone the repo and will copy the src folder from the repo to the package.

#### Alternative Manual Installation
You should be able to navigate into the folder and then pull the repo to update it.
```
cd ai-voice-cloning
git pull
```
If there are large features added, you may need to delete the venv and the re-run the setup-cuda script to make sure there are no package issues

### Linux via Docker
You should be able to navigate into the folder and then pull the repo to update it, then rebuild your Docker image.
```
cd ai-voice-cloning
git pull
./setup-docker.sh
```

## Documentation

### Troubleshooting Manual Installation
The terminal is your friend.  Any errors or issues will pop-up in the terminal when you go to try and run, and then you can start debugging from there.
- If somewhere in the process, torch gets messed up, you may have to reinstall it.  You will have to uninstall it, then reinstall it like the following.  Make sure to type (Y) to confirm deletion.

```
.\venv\Scripts\activate.bat
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Bug Reporting

If you run into any problems, please open up a new issue on the issues tab.

## Tips for developers
`setup-cuda.bat` should have everything that you need for the packages to be installed.  All of the different requirements files make it quite a mess in the script, but each repo has their requirements installed, and then at the end, the `requirements.txt` in the root is needed to change the version *back* to compatible versions for this repo.
