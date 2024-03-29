# Changelogs & Notes

## Implementations and Ideas for Speeding up Tortoise
GPT2:
- https://github.com/152334H/tortoise-tts-fast/issues/3

AR Quantization
- int4? - https://github.com/neonbjb/tortoise-tts/issues/526
- ggml - https://github.com/ggerganov/ggml/issues/59
- TortoiseCPP https://github.com/balisujohn/tortoise.cpp

## 3/29/2024
Additional clean-up, discovered that some key files were missing from the repo if you were going to do a manual install
- setup-cuda.bat is much more handsoff now, it pretty much downloads everything you need.

Bug/error:
- Max recursion depth met when trying to use a chunk size of 20... results in training not occuring. Lowered chunk size to 15 by default, not sure why this occurs, but it's possible too long of audio files can't get processed and end up in an endless recursive loop.

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