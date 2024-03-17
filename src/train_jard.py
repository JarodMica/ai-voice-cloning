import os
import sys
import argparse
import yaml
import datetime

if os.path.exists("runtime"):
	# Get the directory where the script is located
	script_dir = os.path.dirname(os.path.abspath(__file__))

	# Add this directory to sys.path
	if script_dir not in sys.path:
		sys.path.insert(0, script_dir)

from torch.distributed.run import main as torchrun

# this is effectively just copy pasted and cleaned up from the __main__ section of training.py
def train(config_path, launcher='none'):
    opt = option.parse(config_path, is_train=True)

    if launcher == 'none' and opt['gpus'] > 1:
        return torchrun([f"--nproc_per_node={opt['gpus']}", "./src/train.py", "--yaml", config_path, "--launcher=pytorch"])

    trainer = tr.Trainer()
    if launcher == 'none':
        opt['dist'] = False
        trainer.rank = -1
        if len(opt['gpu_ids']) == 1:
            torch.cuda.set_device(opt['gpu_ids'][0])
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        tr.init_dist('nccl', timeout=datetime.timedelta(seconds=5*60))
        trainer.world_size = torch.distributed.get_world_size()
        trainer.rank = torch.distributed.get_rank()
        torch.cuda.set_device(torch.distributed.get_rank())

    trainer.init(config_path, opt, launcher, '')
    trainer.do_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, help='Path to training configuration file.', default='./training/voice/train.yml', nargs='+') # ugh

    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='Job launcher')
    args = parser.parse_args()
    args.yaml = " ".join(args.yaml) # absolutely disgusting
    config_path = args.yaml
    # config_path = "./training/vietnamese/train.yaml"

    with open(config_path, 'r') as file:
        opt_config = yaml.safe_load(file)

    # yucky override
    if "bitsandbytes" in opt_config and not opt_config["bitsandbytes"]:
        os.environ['BITSANDBYTES_OVERRIDE_LINEAR'] = '0'
        os.environ['BITSANDBYTES_OVERRIDE_EMBEDDING'] = '0'
        os.environ['BITSANDBYTES_OVERRIDE_ADAM'] = '0'
        os.environ['BITSANDBYTES_OVERRIDE_ADAMW'] = '0'

    try:
        import torch_intermediary
        if torch_intermediary.OVERRIDE_ADAM:
            print("Using BitsAndBytes optimizations")
        else:
            print("NOT using BitsAndBytes optimizations")
    except Exception as e:
        pass

    import torch
    from dlas import train as tr
    from dlas.utils import util, options as option

    train(config_path, args.launcher)