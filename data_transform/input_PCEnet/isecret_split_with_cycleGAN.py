"""
cycelGan 폴더 안에 있는 것 대로 가져오기 
----------------------------------------
 date: 2023.03.07
 made by: Ellen
 contact: jieunoh@postech.ac.kr

"""

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import shutil


# 0. augment 설정 ------------------

# MIV >>>>>>>>>>>
### PCE ingredient ### 
# for training
rhq_img_path = "/root/jieunoh/ellen_data/ingredient_pce_hq/image"
rhq_mask_path = "/root/jieunoh/ellen_data/ingredient_pce_hq/mask"
degraded_path = "/root/jieunoh/ellen_data/ingredient_pce_hq/degradation_image"

# for testing
rlq_img_path = "/root/jieunoh/ellen_data/ingredient_pce_lq/image" 
rlq_mask_path= "/root/jieunoh/ellen_data/ingredient_pce_lq/mask"

### base cycgan dataset ### 
base_dataset = "/root/jieunoh/ellen_data/0_cyclegan_input_isecretversion_setnew/0_cyclegan_input_isecretversion_setnew_3"
cgfolder = ["trainB","valB","valA"] # test와 val 다른데, pce에서는 val을 사용안함


### output ### 
gen_path="/root/jieunoh/ellen_data/0_pce_input_isecretversion_setnew_3"
pcfolder =["source_gt","target_gt","target_image"]


genimageformat="png" #생성 할 이미지 포멧/원본 이미지 포멧
imageformat="jpeg" # 원본 이미지 포멧
#
# 00. seting 된 parameter 확인  ----------------------------------------------------------------------------------------- 

print("path check")
print("for training, rhq & degraded ---")
print("rhq_img_path: ", rhq_img_path)
print("rhq_mask_path: ", rhq_mask_path)
print("degraded_path: ", degraded_path)
print("for testing, rlq ---")
print("rlq_img_path: ", rlq_img_path)
print("rlq_mask_path: ", rlq_mask_path)


print("원본 이미지 포멧: ", imageformat, ",",genimageformat)
print("생성할 이미지 포멧: ",genimageformat)


print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게

# 0-1. path없으면 path생성 ------------------------------------------------------------------------------------------
print("train---")
if not os.path.isdir(gen_path+"/source_gt"):
    os.makedirs(gen_path+"/source_gt")
    print("source_gt(hq) path generated!")
if not os.path.isdir(gen_path+"/source_image"):
    os.makedirs(gen_path+"/source_image")
    print("source_image(degraded) path generated!")
if not os.path.isdir(gen_path+"/source_mask"):
    os.makedirs(gen_path+"/source_mask")
    print("source_mask path generated!")
print("test---")
if not os.path.isdir(gen_path+"/target_gt"):
    os.makedirs(gen_path+"/target_gt")
    print("target_gt(hq) -aren't going to use- path generated!")
if not os.path.isdir(gen_path+"/target_image"):
    os.makedirs(gen_path+"/target_image")
    print("target_image(lq) path generated!")
if not os.path.isdir(gen_path+"/target_mask"):
    os.makedirs(gen_path+"/target_mask")
    print("target_mask(lq) path generated!")
if not os.path.isdir(gen_path+"/target_gt_mask"):
    os.makedirs(gen_path+"/target_gt_mask")
    print("target_gt_mask(hq) path generated!")

source_gt=0
source_mask=0
source_image =0
target_gt =0
target_gt_mask=0
target_image=0
target_mask=0

# -------------------------------------------------------------
for img in tqdm(os.listdir(os.path.join(base_dataset,cgfolder[0]))): 
    #trainB - source_gt, source_image(-0.png), source mask
    if imageformat in img:
        # img
        source_path = os.path.join(rhq_img_path,img)
        copy_path = os.path.join(gen_path,"source_gt",img)
        shutil.copy(source_path,copy_path)
        # mask
        source_path = os.path.join(rhq_mask_path,img)
        copy_path = os.path.join(gen_path,"source_mask",img)
        shutil.copy(source_path,copy_path)

        # degrade
        source_path = os.path.join(degraded_path,img.split(".")[0]+"-0.png")
        copy_path = os.path.join(gen_path,"source_image",img.split(".")[0]+"-0.jpeg")
        shutil.copy(source_path,copy_path)

for img in tqdm(os.listdir(os.path.join(base_dataset,cgfolder[1]))): 
    #valB - target_gt, traget_gt_mask
    if imageformat in img:
        # img
        source_path = os.path.join(rhq_img_path,img)
        copy_path = os.path.join(gen_path,"target_gt",img)
        shutil.copy(source_path,copy_path)
        # mask
        source_path = os.path.join(rhq_mask_path,img.split(".")[0]+".jpeg")
        copy_path = os.path.join(gen_path,"target_gt_mask",img)
        shutil.copy(source_path,copy_path)

for img in tqdm(os.listdir(os.path.join(base_dataset,cgfolder[2]))): 
    # valA - target_image, target_mask
    if imageformat in img:
        # img
        source_path = os.path.join(rlq_img_path,img)
        copy_path = os.path.join(gen_path,"target_image",img)
        shutil.copy(source_path,copy_path)
        # mask
        source_path = os.path.join(rlq_mask_path,img)
        copy_path = os.path.join(gen_path,"target_mask",img)
        shutil.copy(source_path,copy_path)