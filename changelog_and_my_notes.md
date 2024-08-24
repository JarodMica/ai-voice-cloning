# Changelogs & Notes

## Implementations and Ideas for Speeding up Tortoise
GPT2:
- https://github.com/152334H/tortoise-tts-fast/issues/3

AR Quantization
- int4? - https://github.com/neonbjb/tortoise-tts/issues/526
- ggml - https://github.com/ggerganov/ggml/issues/59
- TortoiseCPP https://github.com/balisujohn/tortoise.cpp

## 8/24/2024
Haven't done work here in awhile, with that being the case, I've forgotten quite a bit of the things that I was working on
- The purpose of this repo was to explore the idea of expanding the embedding table for vocabulary in tortoise, this is very much possible and how I did it is in expand_tortoise.py
- This branch removes the clvp model and allows for 1 sample inference (the updates are done in the tortoise repo not this one though)
- Looking at the changes, added advanced settings inside of the webui

## 5/3/2024
### Oops
- I've been training on the multilingual dataset for the past few days... and I forgot to convert the train.txt kanji into hiragana... oops.
    - I had converted a train.txt file, I just forgot to replace it -_-, wasted time. *sigh*

### Updates
- Added dynamic number_text_token loading to stop having to manually change dimensionality in api.py.  Allows you to load the AR model without the code complaining.

## 5/1/2024
### Memory Leak Issue Pinpointed
- An issue with whisperx processes occur when after running whisperx, "python.exe" processes are **not** killed due to a memory acces violation error.  IDK what caused it exactly, but this is an issue with **my machine** and not any library or code in specific.  These processes show as "suspended" in the resource monitor
    - The full error from Windows Event Viewer is:
    ```
    Faulting application name: python.exe, version: 3.11.8150.1013, time stamp: 0x65c2ad47
    Faulting module name: nvcuda64.dll, version: 31.0.15.5222, time stamp: 0x661852f9
    Exception code: 0xc0000005
    Fault offset: 0x00000000001a0660
    Faulting process id: 0x0xEBC
    Faulting application start time: 0x0x1DA9C4527B7C7A1
    Faulting application path: C:\Users\jarod\AppData\Local\Programs\Python\Python311\python.exe
    Faulting module path: C:\Windows\system32\DriverStore\FileRepository\nv_dispsig.inf_amd64_defcd1ccab02e3ec\nvcuda64.dll
    Report Id: d8d49e6a-e803-494d-9d82-d52d57660f15
    Faulting package full name: 
    Faulting package-relative application ID: 
    ```
- Reasons I think it's my computer:
    1. It was working just fine a few weeks ago, and I was not getting OOM issues on CPU RAM
    2. The package of AVC-3.0 version works just fine on my 3060 alt computer, but not my main PC.  I also don't get this error in the event viewer on my 3060
    3. I recently updated nvidia drivers to get AVC working with Docker in WSL
        - I needed drivers to be updated for Docker to work, meaning it's possible I borked something

That being said, I have tried a few things in order to resolve this issue from:
1. Reinstalling nvidia drivers (and using DDU) and trying other versions
2. Trying out various pytorch versions (as I thought it was a pytorch issue at first)
3. Changing the code to spawn a process which should be killed after running whisperx, but the implementation didn't work as planned

I was reading somewhere that I might just need to reinstall windows to a stable point or reinstall windows, which... I'm not opposed to doing, but, until I get to it... I created a bandage script that:
- Kills ```python.exe``` processes that have a working set size of ```47kb < x < 100kb``` as I found that most of these processes are either 48kb, 52kb, or 56kb.  It just has to be ran as an admin.
- Definetly not a long term solution, but it works for now

## 4/30/2024
### Memory Leak Issue
- Not able to process lots of files for some reason... I thought the below was the issue, but turns out, it is not.
#### ~~Whisperx Memory caused by pytorch 2.2.2~~
- Found a bug where memory leak occuring when using whisperx on pytorch 2.2.2.  This causes out of memory issues for RAM after running whisperx a lot of times and fills up RAM.  Only can be cleared by restarting computer.
    - Error is: RuntimeError: mkl_malloc: failed to allocate memory
        - freezing pytorch at 2.2.1 does NOT seem to resolve the issue for this repo atm
    - For Python 3.11+, I think the issue is related to: https://github.com/pytorch/pytorch/issues/119607


## 4/25/2024
### Note for training on extended model
- First try for training with a larger tokenizer and it was a disaster.  So I think I messed up on a few things:
    - Trying to create a large tokenizer by training on the original tortoise tokenizer and a new, 256 japanese tokenizer
        - This will not work.  Naively, I ran training with it, but essentially, I have jumbled all of the token orders, including the tokens from the orginal tokenizer, which makes freezing the tokens completely irrelevent.  So old token values are now misaligned, and since those token indexes are "frozen", they don't train... and on inference, they don't have their original tokenizer indexs so it produces nonsensical outputs.
    - What I need to do is **add** tokens to the original tokenizer to see if this will work, and train on this.
        - The purpose is I'm still trying to figure out if by freezing tokens, I can "iteratively add" new langauges.  If so, each time that I need to add a new language, I don't have to rerun the entire training on the concatenated dataset; I simply need to extend it, add tokens to the current tokenzizer, and run training.
- Second, tried additional training runs with the tokenzier just appended to the original tokenzier and am not able to maintain the weights of the base tortoise model
    - After a few runs, I have a suspicion that I'm doing something wrong and my understanding isn't deep enough to piece all of the parts together
    - My naive assumption is that I can set requires_grad=False for text embeddings for the original weights, copy them into a new model with a larger embedding table, train with a new language and larger tokenizer, and voila; a multilingual model.
        - It's not that simple though, it's training all weights, so the english goes away...
        - I think it would be just fine if I trained it on a multilingual dataset, so maybe I'll try that first before coming back to this approach.


## 4/24/2024
### Note in general
- Training a new GPT2 model can be done by removing (comment out) "pretrain_model_gpt: './models/tortoise/autoregressive.pth'" from the train yaml files
- Noticed that training was going SUPER slow and wasn't saturating the GPU.  In the train.yaml file, n_workers can be modifed to a larger value which can increase the training speed.  By default, it's set at 2, but by increasing it to 16 on my device with a batch size of 32, my utilization was 100% (numbers will vary based on your PC). 
    - The issue seemed to be that the dataloader wasn't loading in enough data to saturate the GPU, therefore, being the bottleneck of the process

### Notes for extending tortoise model
- The first approach I'm taking with extending the AR is **copying and freezeing** the current weights of the model.  This way, I can hopefully still have the learned features from the original base training and we don't get "catastrophic forgetting" for existing tokens.
    - The first test I did was copying the weights and inferring with the same tokenizer to make sure the outputs are still the same.  With this, my intuition leads me to say that there were no issues with extending the embeddings so it should be good for the next step.
- The text embeddings can be extended for the AR in tortoise from 256 to xxx, but in doing so, we run into issues with the architecture/code:
    CLVP issues:
        - CLVP needs to be retrained or extended itself (I'd assume).  By default, it is also trained with a num_text_tokens of 256 which was determined by the original author/trainer.  Because this is the case, it causes "index out of range" issues when trying to take in the embeddings of the "extended" AR model
        - CLVP needs to be removed from the pipeline.  This is the approach I'm going with as I don't want to train a new CLVP model, AND in my usage of tortoise, the outputs are fine without much influence from the CLVP (as the CLVP doesn't work or have much of an effect with other languages presumably, one area I've been testing out a lot)
            - Bonus is that this make it just a *tad* faster, but not by much.  We, in essence, don't need to compute more than **1 sample** with the AR and are simply relying on the first output of the AR model instead of the rankings from CLVP
    AR output clipping:
        - The AR output needs to be clipped or else it will output 23 second long output files with possibly lots of silence.  This is already handled by fix_autoregressive_output(), so it just needs to be moved out of the CLVP block


## 3/29/2024
Additional clean-up, discovered that some key files were missing from the repo if you were going to do a manual install
- setup-cuda.bat is much more handsoff now, it pretty much downloads everything you need.

Bug/error:
- Max recursion depth met when trying to use a chunk size of 20... results in training not occuring. Lowered chunk size to 15 by default, not sure why this occurs, but it's possible too long of audio files can't get processed and end up in an endless recursive loop.
    - I think the issue lies in the parameter "max_wav_length", which, can actually be adjusted to accept larger audio files.  Too large and you run into errors, but 26 second max seems to be training.  I wonder if this will help with longer sentences... Regardless, something to look into maybe

## 3/24/2024
Just cleaning up somethings and running tests on the code to make sure it functions as it should.  I should think of maybe a way to automate this... but that's a problem for another time.
- Some values like number of processes (num_processes) to spawn based on your CPU cores added for conversion tasks  
- Changed tab "Prepare Other Langauge" to "Prepare Dataset for Large Files" 
- Moved all of the imports inside of main.py into the __name__ check to reduce overhead of multiprocessing
- Ironing out continuation of transcription in case interrupted, so far, the cases I've tested I've fixed and added approrpiate code to accomodate these situations.  The only test case that doesn't work correctly would be if a file is interrupted in the middle of splitting segments based on the srt script since the segments never get written to train.txt...
    - Maybe have a way of mapping what has already been segmented to the srt file that exists there? I'll have to think about this one. 
    Other stuff
        - Removes the "temp" file that is created for rename
        - Modified the dataset script maker to ignore folders that contain mp3 segments already

## 3/23/2024
- Comment out valle and bark instantiations to clean up console

## 3/22/2024
Completed initial implementation of "Prepare Other Language" gradio tab.  A lot of "polishing" going on right now, simply, just going through the app, step by step, and making sure where bugs occur due to upgrades, that those are resolved
- Enables easy usage of transcribing a dataset, and then creating a BPE tokenizer for it
    - All of the pathing for the most part is handled "smartly"

## 3/20/2024
Updating old dependencies and migrating to python 3.11.  This was spurred on because gradio did not want to output an error message no matter what I tried, so I figured why not update to the latest and greatest 🙃🤔
- Deepspeed 14.0 built and installed for python 3.11, along with pytorch 2.2.1+cuda121. Build instruction credits go to https://github.com/S95Sedan/Deepspeed-Windows
    - This is needed to allow for pydantic 2.6.4.  Since the latest version of gradio requires it... well I had to upgrade this.  Luckily 14.0 works with it
- Fairseq 0.12.4 built for python 3.11 using this repo: https://github.com/VarunGumma/fairseq.  There is an issue with the official repo talked about here: https://github.com/facebookresearch/fairseq/issues/5012
- gradio updated to 4.22.0.
    - Particularly time consuming to update as several things were removed from gradio 3 to 4, so updated those lines of code
        - Several functions from gradio.utils moved --> gradio.analytics
        - concurrency_count comment out for the ui as this is a deprecated feature.  Haven't removed it yet...
        - dropdown.update --> you don't need update anymore, just pass back the dropdown menu
- Hifigan fixes:
    - a "squeeze" issue for the output with hifigan fixed in utils.py
    - Removed unsued values in the kwargs passed.  hf transformers doesn't like it.  Ideally, best to never have even set them or stored them if hifigan is selected, but this will work for now


## 3/17/2024
- Adding in other language capability training to the repo, a few files are modifed in dlas and tortoise-tts modules for the cleaners to allow this to happen.
- In both DLAS and Tortoise-tts, update the preprocessing text to be able to handle other languages
- Added https://github.com/JarodMica/tortoise-dataset-tools inside of modules (will be used for other languages)
    - In this case, made whisperx a mandatory install
    - Adding a new tab called right now "Prepare Other Langauge" that will allows you to use the dataset tools I used for other languages

## 2/11/2024
- Allow for decimal values in the epoch text box as a bandaid to the async gradio issue, causing the training run to crash.  Not sure if it's gradio or if it's an error in the DLAS trainer, but this will need to be fixed as it's quite annoying to have to restart training over and over
    - Only seems to happen on very BIG datasets... (JP corpus)

## 1/30/2024
There's something that has been bugging me with whisperx and dataset curation for Japanese; that is, the whisper.json looks completely fine, yet, there are only a few lines inside of train.txt and validation.txt.  Well, looking through the dataset code, it seems that by default, the trainer skips segmentations that are longer than "MAX_TRAINING_DURATION = 11.6097505669" which is a global variable in utils.py and shorter than "MIN_TRAINING_DURATION = 0.6". 
- Tested that this was the case by increasing it to an arbitrarily large max duration and the train.txt was filled with all the files segmented
    - Another note is you don't need to re-whisper files if they have already been processed, just use the (Re)Create Dataset button

This may only be an issue with Japanese as Kanji count as a single character, yet may have a reading of 6 equivalent romaji characters
- There is a setting in whisperx that determines this which is chunk_size ~ info can be found here https://github.com/m-bain/whisperX/pull/445
- Fixed the line where the langauge parameter wasn't being passed to the whisperx transcribe function call, therefore, causing it to auto-deduce the lanugage instead.  

## 1/25/2024
Started working on CPU inference capabilities.  I don't think training is even a thing on CPU so not even going to try
- CPU enabled for inferencing, works with RVC on/off
    - It's very slow, roughly 5-10x slower than GPU but it works
    - Causes for this are lack of deepspeed compatibility on windows
- Hifigan DOES NOT work with CPU inferencing ATM.  Not sure what the issue is or what is causing it, so I'm trying to figure that out
- Currently looking to see if there are options to speed up CPU inference. 
    - BetterTransformers via optimum - didn't notice any difference here, could be doing it wrong
    - Deepspeed for windows - requires a linux OS to my research.  You have to do some type of intel for pytorch / deepspeed install and there are additional pieces that have wheels only built for linux.   

## 1/15/2024
- Manual installation of this with RVC will be quite the hassle due to the assets folder, so what I'll do is put that on HF so that can be downloaded and put into the rvc folder

## 1/14/2024
RVC Added to the repository for inference
- Look at 48khz hifigan
- NOTE TO SELF, when installing rvc-tts-pipeline at this time, you modified the rvc_infer to look for the rvc package inside of modules instead of having it in the parent folder.
    - Also modified imports in rvc: find and replace all rvc.infer. with modules.rvc.infer.
    - Modified hardcoded paths in rvc configs
    - utils.py for hubert and target_folder
    - pipeline.py for rmpve path
- Add option to be able to use RVC voice model
    - EXEC_SETTINGS used instead as we just need to set a parameter to be able to use it
- Add a function that handles rvc voice models for the drop down menu.  Voice models are located at models/rvc_models
- Add rvc voice model refresh to update_voices and the refresh_voices button
- Addtional Note to self: RTFM
    - Spent a lot of time (3+ hours) trying to figure out how to deal with the event handling for sliders because it was trying to call too many events, too fast, making the value not save in rvc.json.  Turns out, easy fix by putting event listening to release instead of change
- Another issue was disabled sliders and inputs.  Use the interactive parameter for gradio elements

## Some date in December
- Added whisper large-v3 to the list of whisper models available.  

## 12/17/2023
- Resolved an import error caused by a newer version of rotary_embedding_torch.
    - Modified portions of the code in dlas to use broadcast_tensors instead of broadcat.  In the latest version of rotary_embedding_torch (0.5.0 > and higher), broadcat was removed due to redudancy as it looks like broadcast_tensors is a part of torch