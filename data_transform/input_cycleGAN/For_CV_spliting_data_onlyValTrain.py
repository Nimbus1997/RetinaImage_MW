"""
 [input_20220623_#_n1000]를 for Cross Validation을 위해서 
 PART 2
 ---PART 1 []----------------------
 1. hq와 lq모우기-> *_hq , *_lq 폴더에 생성 0001부터 - 1000까지

 ---PART 1 [ V ]----------------------
  2. 3 cross val 만들기 -> based "cycleGAN_rlqrhq_input_20220622.py"
    (1) 원래처럼 train-val 순 2:1 (test에도 val과 같은 image 넣음)
    (2) 총 3세트 만들기
----------------------------------
 
 date: 2022.12.10
 update: 2023.01.17
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
# Mediwhale
# rhq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/hq_20220623_512_n1000"
# rlq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/lq_20220623_512_n1000" 
# new_path="/home/guest1/ellen_data/UKB_quality_data2_combined/input_cyclegan/ukb_512"
# imageformat="jpg" #원본 이미지 포멧

# # MIV
rhq_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_hq_512"
rlq_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_lq_512" 
new_path="/root/jieunoh/ellen_data/0_cyclegan_input/cginput_eyeq_total"
imageformat="jpeg" #원본 이미지 포멧


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
print("rlq_path: ", rlq_path)
print("rhq_path: ", rhq_path)
print("num_how_much_more_dataset:", total_dataset)
for path in path_list:
    print("gen_path:", path)

print("원본 이미지 포멧: ", imageformat)

print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게



# 0-1. path없으면 path생성 ------------------------------------------------------------------------------------------
for gen_path in path_list:
    print(gen_path)
    if not os.path.isdir(gen_path+"/trainA"):
        os.makedirs(gen_path+"/trainA")
        print("train path generated!")
    if not os.path.isdir(gen_path+"/valA"):
        os.makedirs(gen_path+"/valA")
        print("val path generated!")
    if not os.path.isdir(gen_path+"/testA"):
        os.makedirs(gen_path+"/testA")
        print("test path generated!")

    if not os.path.isdir(gen_path+"/trainB"):
        os.makedirs(gen_path+"/trainB")
        print("train path generated!")
    if not os.path.isdir(gen_path+"/valB"):
        os.makedirs(gen_path+"/valB")
        print("val path generated!")
    if not os.path.isdir(gen_path+"/testB"):
        os.makedirs(gen_path+"/testB")
        print("test path generated!")

# 0-2. dataset 나누기 -train val test 개수------------------------------------------------------------------------------------------------------------

# low qualtiy -> A group
ltotal_dataset = len(os.listdir(rlq_path))

ltrainN = int(float(ltotal_dataset)/float(trainlR+vallR)*float(trainlR))
lvalN = ltotal_dataset - ltrainN


# high qualtiy -> B group & A group in test
htotal_dataset = len(os.listdir(rhq_path))
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
# 1. real low quality image  불러오기 [A group] -------------------------------
low_quality_block_size=ltotal_dataset/total_dataset
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    val_start = low_quality_block_size*index

    i=0
    val_count=0
    train_count=0

    for rlq in tqdm(sorted(os.listdir(rlq_path))):
        if imageformat in rlq:
            source_path=rlq_path+'/'+rlq

        #2. real lq image 저장하기 ------------------
            # val
            if (i>=val_start) and (i<(val_start+low_quality_block_size)): 
                copy_path_val = path+"/valA/"+rlq 
                copy_path_test=path+"/testA/"+rlq
                shutil.copy(source_path,copy_path_val)
                shutil.copy(source_path,copy_path_test)
                val_count+=1
            
            else:
                copy_path = path+"/trainA/"+rlq           
                shutil.copy(source_path,copy_path)
                train_count += 1
            i += 1
    print("train/val/test: ", str(train_count),"/",str(val_count))
print("[low quality end]==================================\n\n")



# 2. real hq import image [B group] ------------------------------------------------------
high_quality_block_size=htotal_dataset/total_dataset
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    test_start = high_quality_block_size*index

    i=0
    val_count=0
    train_count=0

    for rhq in tqdm(sorted(os.listdir(rhq_path))):
        if imageformat in rhq:
            source_path=rhq_path+'/'+rhq

        #2. real lq image 저장하기 ------------------
            # val
            if (i>=test_start) and (i<(test_start+high_quality_block_size)): 
                copy_path_val=path+"/valB/"+rhq
                copy_path_test=path+"/testB/"+rhq
                shutil.copy(source_path,copy_path_val)
                shutil.copy(source_path,copy_path_test)
                val_count+=1
            
            # train
            else:
                copy_path = path+"/trainB/"+rhq           
                shutil.copy(source_path,copy_path)
                train_count += 1
            i += 1
    print("train/val/test: ", str(train_count),"/",str(val_count))
print("[high quality end]==================================\n\n")
