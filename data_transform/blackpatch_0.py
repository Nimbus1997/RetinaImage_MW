import numpy as np
import cv2 as cv
from PIL import Image
import os
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision.transforms import transforms
# ----------------------------------------------------------------
# making low quality retina image from high quality image
# version 0. same black patch 작은 사이즈로 어떻게 하나 test (v)
# version 1. same black patch
# version 2. random black patch
#           - patch 개수는 1개로 고정, 1) black gaussian sd , 2) patch size -범위 정해서 그 range에서, 3) patch 위치 random

# 2021.12.30 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

# Tensorboard
writer = SummaryWriter('runs/experiment0')

# 1. base image
# imagen= np.random.randint(0,255, size=(3,300,300))
imagen= np.full((3,300,300),255)
imagen1= np.full((300,300),255)

imaget=torch.from_numpy(imagen)
writer.add_image('image',imaget,0)

print("<><>>>",imagen1)
imagei=Image.fromarray(np.uint8(imagen1))
imagei.save("./ori.png")

imagei0=Image.fromarray(imagen, 'RGB')
print(">>>",imagei0.size)
imagei0.save("./ori0.png")


# 2. 2d gaussian filter
kernel1d = cv.getGaussianKernel(100, 5)
print("1d:", kernel1d)
kernel2d = np.outer(kernel1d, kernel1d.transpose())
print(kernel2d.shape)
print()
# 2d gaussian filter 반전(중심이 black되게)
kernel2db = 1.-np.outer(kernel1d, kernel1d.transpose())
# print("____",kernel2db)

# 겉에 pad하기 위해서(나중에 image와 곱하려면 image size로 만들어줘야함) array--> image
imgkernel = Image.fromarray(kernel2db)
# imgkernel.show()
print(imgkernel.size) # 4, 4
# padding
paddkernel = torchvision.transforms.Pad(
    padding=(100,100,100,100), fill=1, padding_mode='constant')(imgkernel)
paddkernelnp = np.array(paddkernel)
print(paddkernelnp)
print(paddkernelnp.shape) # 10, 10  
pad255 = int(255)*paddkernelnp
print(pad255)

kernel_img = Image.fromarray(255*paddkernelnp)
kernel_img.save("./kernel.jpg")
paddkernelto = torch.from_numpy(paddkernelnp)
print(paddkernel.size) # 10, 10 
print(paddkernelto.shape) # 10, 10 

mul = imaget*paddkernelto
trans_ti=torchvision.transforms.ToPILImage()
mul_i = trans_ti(mul)
mul_i.save("./mul_i.png")

print(mul.shape)
# tensorboard
writer.add_image('black gaussian patch result', mul)
writer.add_image('black gaussian patch result', mul)
writer.close()

# result
# print(paddkernelnp)
