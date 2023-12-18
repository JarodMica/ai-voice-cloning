import torch

devices = [f"cuda:{i} => {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]

print(devices)