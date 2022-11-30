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
blackness = 150  # 0-255 (낮을수록 어두움)


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

    white_n = np.full((imageH, imageW, 3), (255, 255, 225), dtype=np.uint8)
    circle_n = cv.ellipse(white_n, (100, 500), (100, 200), 10, 0, 360, (blackness,
                          blackness, blackness), -1)  # (중심위치), (X,Y길이), 회전, 0도부터 360도까지, (색) , -1(채움)
    circleblurr_n = cv.blur(circle_n, (150,200))
    print(circleblurr_n)
    print("circleblurr_n shape:", circleblurr_n.shape)
    mask_n =circleblurr_n[:,:,0]/255.
    print("mask_n shape:", mask_n.shape)
    mask_t =torch.Tensor(mask_n)

    # 3. image에 gaussian black patch add ------------------
    result_t = mask_t * img_t
    writer.add_image(file+"_masked_black", result_t)

    # 4. image 저장 ------------------
    result_i = trans_ti(result_t)
    result_i.save(gen_path+"/circleGblurr_"+file)
