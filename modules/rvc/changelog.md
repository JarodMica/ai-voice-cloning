# Changelog

## 10/6/2023
I removed fairseq from requirements.txt as if you don't have microsoft build tools for c++ 2019, it will fail to install packages.  This can be remedied by either installing the build tools, or using a prebuilt wheels (which is the approach I'm going to use for my applications that I distribute)
- Added weights_only=True to all torch.load calls to help mitigate possible security issues with pickle.
    - I have not tested to see if the package will still work for training with this new parameter, but I will have to test this at a later time when I get to that fuctionality.