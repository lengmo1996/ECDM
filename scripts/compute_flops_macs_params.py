import torch,os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from calflops import calculate_flops
from ecdm.modules.diffusionmodules.simple_unet import SimpleUNet
from ecdm.models.diffusion.ecdm_second_stage import ECDMSecondStage
import yaml

img_rgb_HF=torch.randn(1, 3, 512, 640)
img_ir=torch.randn(1, 3, 512, 640)


with open('configs/ecdm_second_stage.yaml','r') as file:
    config=yaml.safe_load(file)

unet=SimpleUNet(**config['model']['init_args']['unet_config']['init_args'])
config['model']['init_args'].pop('unet_config')
model=ECDMSecondStage(unet,**config['model']['init_args'])

args=[img_ir,img_rgb_HF]
flops, macs,params = calculate_flops(model,output_as_string=True,args=args,forward_mode="generate",
                                      output_precision=4)
print(flops, macs,params)