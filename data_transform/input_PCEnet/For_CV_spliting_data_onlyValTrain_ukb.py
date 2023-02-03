"""
 [input_20220623_#_n1000]를 for Cross Validation을 위해서 
 PART 2
 ---PART 1 []----------------------
 1. hq와 lq모우기-> *_hq , *_lq 폴더에 생성 0001부터 - 1000까지
 2. PCE Net preprocessing, degaradation 

 ---PART 1 [ V ]----------------------
  2. 3 cross val 만들기 -> based "cycleGAN_rlqrhq_input_20220622.py"
    (1) 원래처럼 train-val 순 2:1 (test에도 val과 같은 image 넣음)
    (2) 총 3세트 만들기
    (3) dataset 구성
        i) train: source_gt (real hq) / source_image (gen lq) / source_mask (mask)
        ii) test: target_gt (real_hq -> 사용안함) / target_image (real_lq) / target_mask (lq)
----------------------------------
 
 date: 2023.01.31
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
# # Mediwhale >>>>>>>>>>>
# ### input ### 
# # for training
# rhq_img_path = "/home/guest1/ellen_code/PCENet-Image-Enhancement/data/get_low_quality/image/high_quality_image_pre_process/image"
# rhq_mask_path = "/home/guest1/ellen_code/PCENet-Image-Enhancement/data/get_low_quality/image/high_quality_image_pre_process/mask"
# degraded_path = "/home/guest1/ellen_code/PCENet-Image-Enhancement/data/get_low_quality/image/low_quality_image"

# # for testing
# rlq_img_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_pcenet/target_image" 
# rlq_mask_path= "/home/guest1/ellen_data/UKB_quality_data2_combined/input_pcenet/target_mask"


# ### output ### 
# new_path="/home/guest1/ellen_data/UKB_quality_data2_combined/input_pcenet/ukb_512"


# MIV >>>>>>>>>>>
### input ### 
# for training
rhq_img_path = "/home/guest1/ellen_code/PCENet-Image-Enhancement/data/get_low_quality/image/high_quality_image_pre_process/image"
rhq_mask_path = "/home/guest1/ellen_code/PCENet-Image-Enhancement/data/get_low_quality/image/high_quality_image_pre_process/mask"
degraded_path = "/home/guest1/ellen_code/PCENet-Image-Enhancement/data/get_low_quality/image/low_quality_image"

# for testing
rlq_img_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_pcenet/target_image" 
rlq_mask_path= "/home/guest1/ellen_data/UKB_quality_data2_combined/input_pcenet/target_mask"


### output ### 
new_path="/root/jieunoh/ellen_data/0_pce_input/ukb_512"



imageformat="png" #원본 이미지 포멧
total_dataset=3

# 0. CV 개수에 따라서 다르게
path_list = []
for num in range(total_dataset):
    numm=num+1
    path_list.append(new_path+"_"+str(numm))

trainlR, vallR = 2,1  # low quality data의 train : val: test 비율 쓰기
trainhR, valhR = 2,1# high qualtiy data의 train : val: test 비율 쓰기

# 00. seting 된 parameter 확인  ----------------------------------------------------------------------------------------- 

print("path check")
print("for training, rhq & degraded ---")
print("rhq_img_path: ", rhq_img_path)
print("rhq_mask_path: ", rhq_mask_path)
print("degraded_path: ", degraded_path)
print("for testing, rlq ---")
print("rlq_img_path: ", rlq_img_path)
print("rlq_mask_path: ", rlq_mask_path)

print("num_how_much_more_dataset:", total_dataset)


for path in path_list:
    print("gen_path:", path)

print("원본 이미지 포멧: ", imageformat)

print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게



# 0-1. path없으면 path생성 ------------------------------------------------------------------------------------------
for gen_path in path_list:
    print(gen_path)
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
        

# 0-2. dataset 나누기 -train val test 개수------------------------------------------------------------------------------------------------------------

# low qualtiy -> A group
ltotal_dataset = len(os.listdir(rlq_img_path))
ltrainN = int(float(ltotal_dataset)/float(trainlR+vallR)*float(trainlR))
lvalN = ltotal_dataset - ltrainN


# high qualtiy -> B group & A group in test
htotal_dataset = len(os.listdir(rhq_img_path))
htrainN = int(float(htotal_dataset)/float(trainhR+valhR)*float(trainhR))
hvalN = htotal_dataset - htrainN



print("\n[A group] ratio of low quality dataset (train : val) : (", trainlR, ": ", vallR, ")")
print("Low qualtiy # of dataset in (train : val) : (",
      ltrainN, ", ", lvalN, ")")


print("\n[B group] ratio of high quality dataset (train : val) : (",
      trainhR, ": ", valhR, ")")
print("High qualtiy # of dataset in (train : val) : (",
      htrainN, ", ", hvalN,")")
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게



# LQ
# 1. real low quality image  불러오기 [A group] only test-------------------------------
low_quality_block_size=ltotal_dataset/total_dataset
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    val_start = low_quality_block_size*index

    i=0
    val_count=0
    train_count=0

    for rlq in tqdm(sorted(os.listdir(rlq_img_path))):
        if imageformat in rlq:
            source_path_img=rlq_img_path+'/'+rlq
            source_path_mask=rhq_mask_path+'/'+rlq# have the same name as the image

        #2. real lq image 저장하기 ------------------
            # val= test (target)
            if (i>=val_start) and (i<(val_start+low_quality_block_size)): 
                copy_path_img = path+"/target_image/"+rlq 
                copy_path_mask=path+"/target_mask/"+rlq
                shutil.copy(source_path_img,copy_path_img)
                shutil.copy(source_path_mask,copy_path_mask)
                val_count+=1
            
            i += 1
    print("train/val=test: ", str(train_count),"/",str(val_count))
print("[low quality end]==================================\n\n")



# 2. real hq import image [B group] ------------------------------------------------------

degraded_set=["0", "1"]

high_quality_block_size=htotal_dataset/total_dataset
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    test_start = high_quality_block_size*index

    i=0
    val_count=0
    train_count=0

    for rhq in tqdm(sorted(os.listdir(rhq_img_path))):
        if imageformat in rhq:
            source_path_img=rhq_img_path+'/'+rhq
            source_path_mask=rhq_mask_path+'/'+rhq # have the same name as the image

            name=rhq.split(".")[0]

        #2. real lq image 저장하기 ------------------
            # val = test (target_gt -> 안쓸예정)
            if (i>=test_start) and (i<(test_start+high_quality_block_size)): 
                copy_path_val_image=path+"/target_gt/"+rhq
                copy_path_val_mask=path+"/target_gt_mask/"+rhq
                shutil.copy(source_path_img,copy_path_val_image)
                shutil.copy(source_path_mask,copy_path_val_mask)

                val_count+=1
            
            # train (source)
            else:
                copy_path_img = path+"/source_gt/"+rhq  
                copy_path_mask = path+"/source_mask/"+rhq
                shutil.copy(source_path_img,copy_path_img)
                shutil.copy(source_path_mask,copy_path_mask)

                for degarde in degraded_set:
                    source_degrade_path = degraded_path+"/"+name+"-"+degarde+"."+imageformat
                    copy_degrade_path = path+"/source_image/"+name+"-"+degarde+"."+imageformat
                    shutil.copy(source_degrade_path,copy_degrade_path)

                train_count += 1
            i += 1
    print("train/val=test: ", str(train_count),"/",str(val_count))
print("[high quality end]==================================\n\n")
