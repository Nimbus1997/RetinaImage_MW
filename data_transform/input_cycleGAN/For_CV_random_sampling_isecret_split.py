"""

 
 date: 2023.03.03
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
import random


# 0. augment 설정 ------------------
# Mediwhale
# rhq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/hq_20220623_512_n1000"
# rlq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/lq_20220623_512_n1000" 
# new_path="/home/guest1/ellen_data/UKB_quality_data2_combined/input_cyclegan/ukb_512"
# imageformat="jpg" #원본 이미지 포멧

# # MIV
rhq_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_hq_512"
rlq_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_lq_512" 
new_path="/root/jieunoh/ellen_data/0_cyclegan_input_isecretversion_set"
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

# low  _ A
n_low_train = 1698
n_low_val = 8471  # test
n_low_test = 178

# high_B
n_high_train = 7482
n_high_val = 4558  #test
n_high_test = 865




print("\n[A group] ratio of low quality dataset (train : val : test) : (", n_low_train, ": ", n_low_val, ": ",n_low_test ,")")


print("\n[B group] ratio of high quality dataset (train : val) :: (", n_high_train, ": ", n_high_val, ": ",n_high_test, ")")
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게



# LQ
# 1. real low quality image  불러오기 [A group] -------------------------------
print("------------------------------------------------------")
low_list = os.listdir(rlq_path)
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")

    random.shuffle(low_list)  #  shuffle!! 
    train_count = 0
    val_count= 0
    test_count=0

    for rlq in tqdm(low_list):
        if imageformat in rlq:
            source_path=rlq_path+'/'+rlq

        #2. real lq image 저장하기 ------------------
            # train
            if (train_count<n_low_train):
                copy_path = path+"/trainA/"+rlq           
                shutil.copy(source_path,copy_path)
                train_count += 1


                copy_path_val = path+"/valA/"+rlq 
                copy_path_test=path+"/testA/"+rlq
                shutil.copy(source_path,copy_path_val)
                shutil.copy(source_path,copy_path_test)
                val_count+=1
            #test
            elif (test_count<n_low_test):
                copy_path_test=path+"/testA/"+rlq         
                shutil.copy(source_path,copy_path)
                test_count += 1
            #val
            else:
                copy_path_val = path+"/valA/"+rlq         
                shutil.copy(source_path,copy_path)
                val_count += 1
    print("train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
print("[low quality end]==================================\n\n")



# 2. real hq import image [B group] ------------------------------------------------------
print("------------------------------------------------------")
high_list = os.listdir(rhq_path)
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")

    random.shuffle(high_list)  #  shuffle!! 
    train_count = 0
    val_count= 0
    test_count=0
    
    for img in tqdm(high_list):
        if imageformat in img:
            source_path=rhq_path+'/'+img

        #2. real lq image 저장하기 ------------------
            # train
            if (train_count<n_low_train):
                copy_path = path+"/trainA/"+img           
                shutil.copy(source_path,copy_path)
                train_count += 1


                copy_path_val = path+"/valA/"+img 
                copy_path_test=path+"/testA/"+img
                shutil.copy(source_path,copy_path_val)
                shutil.copy(source_path,copy_path_test)
                val_count+=1
            #test
            elif (test_count<n_low_test):
                copy_path_test=path+"/testA/"+img         
                shutil.copy(source_path,copy_path)
                test_count += 1
            #val
            else:
                copy_path_val = path+"/valA/"+img         
                shutil.copy(source_path,copy_path)
                val_count += 1
    print("train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
print("[high quality end]==================================\n\n")
