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
# ====>  version 1. same black patch -실제 이미지에 입혀보기(v) - const pad mask 만들고, mask를 gaussian blur사용하기
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
    print("============================")
    
    img = Image.open(ori_path+"/"+file)
    img_t = trans_it(img)
    imageH, imageW = img.size
    print("image height and width: (", imageH, ", ", imageW, ")")
    print("image.shape: ", img_t.shape)
    writer.add_image(file+"_ori", img_t)

    # 2. Gaussian black filter 만들기 ------------------
    kernel_size = 500
    # black_n = np.full((kernel_size,kernel_size),0) # type1: 완전 검정
    black_n = np.full((kernel_size, kernel_size), 0.5)  # type2: 반정도 어둡게
    print(">> black_n: ", black_n)

    pad_lr = int((imageW-kernel_size)/2)
    pad_tb = int((imageH-kernel_size)/2)
    # balck_i = Image.fromarray(np.uint8(black_n)) # type1: 완전 검정
    balck_i = Image.fromarray(black_n)  # type2: 반정도 어둡게
    # 이건 출력하면 value값이 나오지 않고, PIL.Image.Image image mode=F size=500x500 at 0x7FAD604DE910 이렇게 나옴
    print(">> balck_i: ", balck_i)

    # padding = (left, top, right, bottom) or (left/right, top/bottom)
    mask_i = torchvision.transforms.Pad(
        padding=(pad_lr, pad_tb), fill=1, padding_mode='constant')(balck_i)
    print("mask_i size:", mask_i.size)

    mask_n = np.array(mask_i)
    print("mask_n shape:",mask_n.shape)
    maskblur_i = cv.GaussianBlur(mask_n,(99,99),0)
    maskblur_t = trans_it(maskblur_i)

    # 3. image에 gaussian black patch add ------------------
    result_t = maskblur_t * img_t
    writer.add_image(file+"_masked_black", result_t)

    # 4. image 저장 ------------------
    result_i = trans_ti(result_t)
    result_i.save(gen_path+"/constGblurr_"+file)
