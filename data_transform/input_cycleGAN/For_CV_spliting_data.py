"""
 [input_20220623_#_n1000]를 for Cross Validation을 위해서 
 PART 2
 ---PART 1 []----------------------
 1. hq와 lq모우기-> *_hq , *_lq 폴더에 생성 0001부터 - 1000까지

 ---PART 1 [ V ]----------------------
  2. 5 cross val 만들기 -> based "cycleGAN_rlqrhq_input_20220622.py"
    (1) 원래처럼 train- val -test 순 [이미 완료] 6:2:2 / 6:2:2
    (2) 4 세트 더 만들기
    (3) low qulaity test set에 hq 넣는 것 빼고 6:2:1:1 --> 6:2:2
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
rhq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220623_512_n1000_hq"
rlq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220623_512_n1000_lq" 
new_path="/home/guest1/ellen_data/UKB_quality_data2_combined/inpt_20230118_512"
imageformat="jpg" #원본 이미지 포멧

# # MIV
# rhq_path = "/root/jieunoh/ellen_data/isecret_input_eyeq_total_hq"
# rlq_path = "/root/jieunoh/ellen_data/isecret_input_eyeq_total_lq" 
# new_path="/root/jieunoh/ellen_data/input_eyeq_total_spilt_new"
# imageformat="jpeg" #원본 이미지 포멧


num_how_much_more_dataset=4

# 0. CV 개수에 따라서 다르게
path_list = []
for num in range(num_how_much_more_dataset):
    numm=num+2
    path_list.append(new_path+"_"+str(numm))

# total image개수 정하고 싶으면------------
set_total = False
total_img_sample = 1000 #l, h each 
# ------------------------



trainlR, vallR, testlR = 6, 2, 2  # low quality data의 train : val: test 비율 쓰기
trainhR, valhR, testhR  = 6, 2, 2 # high qualtiy data의 train : val: test 비율 쓰기
biggerthanthis = 10000  # 10000보단 작을 것으로 예상
sw = biggerthanthis  # smallst widht 젤작은 width 알아보기 위해서
sh = biggerthanthis  # smallest height 젤 작은 hieght 알아보기 위해서

# 00. seting 된 parameter 확인  ----------------------------------------------------------------------------------------- 

print("path check")
print("rlq_path: ", rlq_path)
print("rhq_path: ", rhq_path)
print("num_how_much_more_dataset:", num_how_much_more_dataset)
for path in path_list:
    print("gen_path:", path)

print("원본 이미지 포멧: ", imageformat)
print("total_img_sample: ", set_total,",", total_img_sample)

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
if set_total:
    ltotal_dataset=total_img_sample
else:
    ltotal_dataset = len(os.listdir(rlq_path))

ltrainN = int(float(ltotal_dataset)/10.*float(trainlR))
lvalN = int(float(ltotal_dataset)/10.*float(valhR))
ltestN = ltotal_dataset - ltrainN-lvalN

ldigit = len(str(ltotal_dataset)) # 자릿수
print("\n-------------------------------")
print("ldigit: ", ldigit)

# high qualtiy -> B group & A group in test
if set_total:
    htotal_dataset=total_img_sample
else:
    htotal_dataset = len(os.listdir(rhq_path))
htrainN = int(float(htotal_dataset)/10.*float(trainhR))
hvalN = int(float(htotal_dataset)/10.*float(valhR))
htestN = htotal_dataset - htrainN-hvalN

hdigit = len(str(htotal_dataset)) # 자릿수
print("hdigit: ", hdigit)



print("\n[A group] ratio of low quality dataset (train : val : test) : (",
      trainlR, ": ", vallR, ": ", testlR, ")")
print("Low qualtiy # of dataset in (train : val : test) : (",
      ltrainN, ", ", lvalN, ", ", ltestN, ")")


print("\n[B group] ratio of high quality dataset (train : val :test) : (",
      trainhR, ": ", valhR, ": ", testhR, ")")
print("High qualtiy # of dataset in (train : val : test) : (",
      htrainN, ", ", hvalN, ", ", htestN, ")")
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게



# LQ
# 1. real low quality image  불러오기 [A group] -------------------------------
low_quality_block_size=ltotal_dataset/5
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    test_start = low_quality_block_size*index

    if index==0:
        val_start=low_quality_block_size*(4)
    else:
        val_start=low_quality_block_size*(index+1)

    i=0
    test_count=0
    val_count=0
    train_count=0

    for rlq in tqdm(sorted(os.listdir(rlq_path))):
        if imageformat in rlq:
            i += 1
            source_path=rlq_path+'/'+rlq

        #2. real lq image 저장하기 ------------------
            # test
            if (i>=test_start) and (i<(test_start+low_quality_block_size)): 
                copy_path=path+"/testA/"+rlq
                shutil.copy(source_path,copy_path)
                test_count+=1
            # val
            elif (i>=val_start) and (i<(val_start+low_quality_block_size)):
                copy_path = path+"/valA/"+rlq           
                shutil.copy(source_path,copy_path)
                val_count += 1
            # train
            else:
                copy_path = path+"/trainA/"+rlq           
                shutil.copy(source_path,copy_path)
                train_count += 1
    print("train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
print("[low quality end]==================================\n\n")



# 2. real hq import image [B group] ------------------------------------------------------
high_quality_block_size=htotal_dataset/5
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    test_start = high_quality_block_size*index

    if index==0:
        val_start=high_quality_block_size*(4)
    else:
        val_start=high_quality_block_size*(index+1)

    i=0
    test_count=0
    val_count=0
    train_count=0

    for rhq in tqdm(sorted(os.listdir(rhq_path))):
        if imageformat in rhq:
            i += 1
            source_path=rhq_path+'/'+rhq

        #2. real lq image 저장하기 ------------------
            # test
            if (i>=test_start) and (i<(test_start+high_quality_block_size)): 
                copy_path=path+"/testB/"+rhq
                shutil.copy(source_path,copy_path)
                test_count+=1
            # val
            elif (i>=val_start) and (i<(val_start+high_quality_block_size)):
                copy_path = path+"/valB/"+rhq           
                shutil.copy(source_path,copy_path)
                val_count += 1
            # train
            else:
                copy_path = path+"/trainB/"+rhq           
                shutil.copy(source_path,copy_path)
                train_count += 1
    print("train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
print("[high quality end]==================================\n\n")
