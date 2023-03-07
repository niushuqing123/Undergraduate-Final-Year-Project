import os

import cv2
import numpy as np
import torch




# import unet

from net.net_ import *


# from net import *
# import utils
from PIL import Image


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
def keep_image_size_open_rgb(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask



from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor()
])



# from data import *
from torchvision.utils import save_image
from PIL import Image

net=UNet(3).cuda()

weights='params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

# _input=input('please input JPEGImages path:')
# _input=r'data\JPEGImages\0b2e702f90aee4fff2bc6e4326308d50cf04701082e718d4f831c8959fbcda93.png'
# _input=r'data\0b2e702f90aee4fff2bc6e4326308d50cf04701082e718d4f831c8959fbcda93.png'
_input=r'data\JPEGImages\2007_000032.jpg'


img=keep_image_size_open_rgb(_input)
img_data=transform(img).cuda()
img_data=torch.unsqueeze(img_data,dim=0)
net.eval()
out=net(img_data)
out=torch.argmax(out,dim=1)
out=torch.squeeze(out,dim=0)
out=out.unsqueeze(dim=0)
print(set((out).reshape(-1).tolist()))
out=(out).permute((1,2,0)).cpu().detach().numpy()
cv2.imwrite('result/result.png',out)
cv2.imshow('out',out*255.0)
cv2.waitKey(0)

