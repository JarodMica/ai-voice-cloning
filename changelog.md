# Changelogs & Notes

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