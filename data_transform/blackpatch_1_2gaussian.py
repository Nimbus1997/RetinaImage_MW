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
# version 0. same black patch 작은 사이즈로 어떻게 하나 test
# ====> version 1. same black patch -실제 이미지에 입혀보기(v) - 가우시안 필터 사용  --별로임
#       const 하게 patch만들고 그걸 blur 해줘야 겠음.
# version 2. random black patch
#           - patch 개수는 1개로 고정, 1) black gaussian sd , 2) patch size -범위 정해서 그 range에서, 3) patch 위치 random

# 2021.12.30 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

writer = SummaryWriter('runs/experiment1')

# 0. augment 설정 ------------------
ori_path = "./ori_hq"
gen_path = "./fake_lq"

black_patch_gaussian_sd = 1  # 작을수록 더 중심 검정
black_patch_size_ratio = 0.4

trans_it = torchvision.transforms.ToTensor()  # Image -> tensor
trans_ti = torchvision.transforms.ToPILImage()  # tensor -> Image

# 1. import image ------------------
for file in os.listdir(ori_path):
    img = Image.open(ori_path+"/"+file)
    img_t = trans_it(img)
    imageH, imageW = img.size
    print("image height and width: (", imageH, ", ", imageW, ")")
    print("image.shape: ", img_t.shape)
    writer.add_image(file+"_ori", img_t)

    # 2. Gaussian black filter 만들기 ------------------
    kernel_size = 800
    kernel_1d_n = cv.getGaussianKernel(
        kernel_size, 20)  # ( size, standard deviation )
    kernel_2d_n = 1. - 8.*np.outer(kernel_1d_n, kernel_1d_n.transpose())

    print(">> kernel_2d_n:", kernel_2d_n)

    pad_lr = int((imageW-kernel_size)/2)
    pad_tb = int((imageH-kernel_size)/2)
    kernel_2d_i = Image.fromarray(kernel_2d_n)
    print(pad_lr)
    print(pad_tb)

    # padding = (left, top, right, bottom) or (left/right, top/bottom)
    mask_i = torchvision.transforms.Pad(
        padding=(pad_lr, pad_tb), fill=1, padding_mode='constant')(kernel_2d_i)
    print("mask_i size:", mask_i.size)

    mask_n = np.array(mask_i)
    mask_t = torch.from_numpy(mask_n)
    print("mask_n shape:", mask_n.shape)
    print("mask_t shape:", mask_t.shape)

    # 3. image에 gaussian black patch add ------------------
    result_t = mask_t * img_t
    writer.add_image(file+"_masked", result_t)

    # 4. image 저장 ------------------
    result_i = trans_ti(result_t)
    result_i.save(gen_path+"/"+file)
