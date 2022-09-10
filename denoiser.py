from PIL import Image
import torchvision.transforms.functional as TF
import time
import torchvision.transforms as T
import random
import torch
import unet


def get(path):
    image= Image.open(path)
    x = TF.to_tensor(image)
    return x[:,-720:,-1280:]


import cv2
#from ../snap/MIRNetv2/Real_Denoising/Options

from basicsr.models.archs.mirnet_v2_arch import MIRNet_v2

yaml_file = 'MIRNetv2/Real_Denoising/Options/RealDenoising_MIRNet_v2.yml'
import yaml
import torch
import torch.nn as nn
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

weights = 'MIRNetv2/Real_Denoising/pretrained_models/real_denoising.pth'

del x["network_g"]["type"]
model_restoration = MIRNet_v2(**(x['network_g']))

checkpoint = torch.load(weights)
model_restoration.load_state_dict(checkpoint['params'])
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


im = get("../datasets/temple/0001-00752.png0001.png")

noisy_patch = im.unsqueeze(0).cuda()
restored_patch = model_restoration(noisy_patch)
#restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

img= T.ToPILImage()(torch.Tensor(restored_patch).permute([2,0,1]))
img.save(name)



