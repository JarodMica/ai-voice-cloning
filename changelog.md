# Changelogs & Notes

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