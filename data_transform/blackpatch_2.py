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
import random
import time


# ----------------------------------------------------------------
# making low quality retina image from high quality image
# version 0. same black patch 작은 사이즈로 어떻게 하나 test
# version 1. same black patch -실제 이미지에 입혀보기
# ====> version 2. random black patch (v)
#           - version1에서 4번circleGblur기반
#            1) blackness - 얼마나 어둡게 할지 20-150
#            2) ellipse (중심위치-아무곳이나), (X Y길이 - 전체 크기에서 랜덤하게 0.2 - 0.6), 회전
#            3) blur - 얼마나 blur할지 100-800사이
# 2022.01.03 jieunoh@postech.ac.kr
# ----------------------------------------------------------------


# 0. augment 설정 ------------------ + 아래 random 부분도 설정 필요
writer = SummaryWriter('runs/experiment1')

ori_path = "../../ellen_data/UKB_quality_data/high_Q"
gen_path = "/home/guest1/ellen_data/UKB_quality_data/gen_lowQ_byHQ_202220103"

black_patch_gaussian_sd = 1  # 작을수록 더 중심 검정
black_patch_size_ratio = 0.4


trans_it = torchvision.transforms.ToTensor()  # Image -> tensor
trans_ti = torchvision.transforms.ToPILImage()  # tensor -> Image

random.seed(time.time())


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

    # random
    blackness = random.randrange(20, 150)
    eX = random.randrange(0, imageW)  # 중심 위치
    eY = random.randrange(0, imageH)
    eW = random.randrange(int(imageW*0.2), int(imageW*0.6))  # x축 반지름
    eH = random.randrange(int(imageH*0.2), int(imageH*0.6))
    eA = random.randrange(0, 360)  # ellipse 회전정도
    eBlurx = random.randrange(100,500)
    eBlury = random.randrange(100,500)


    white_n = np.full((imageH, imageW, 3), (255, 255, 225), dtype=np.uint8)
    ellipse_n = cv.ellipse(white_n, (eX, eY), (eW, eH), 10, 0, 360, (blackness,
                                                                     blackness, blackness), -1)  # (중심위치), (X,Y길이), 회전, 0도부터 360도까지, (색) , -1(채움)
    ellipseblurr_n = cv.blur(ellipse_n, (eBlurx, eBlury))
    # print(ellipseblurr_n)
    print("ellipseblurr_n shape:", ellipseblurr_n.shape)
    mask_n = ellipseblurr_n[:, :, 0]/255.
    print("mask_n shape:", mask_n.shape)
    mask_t = torch.Tensor(mask_n)

    # 3. image에 gaussian black patch add ------------------
    result_t = mask_t * img_t
    writer.add_image(file+"_masked_black", result_t)

    # 4. image 저장 ------------------
    result_i = trans_ti(result_t)
    result_i.save(gen_path+"/"+file)
