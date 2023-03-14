"""
 To make it ressemble fold 3  -> with the isecret label
 필요 없음 - 원래것도 문제 X (label 같음)
 date: 2023.03.04
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
import pandas as pd

# 0. augment 설정 ------------------
# Mediwhale
# rhq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/hq_20220623_512_n1000"
# rlq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/lq_20220623_512_n1000" 
# new_path="/home/guest1/ellen_data/UKB_quality_data2_combined/input_cyclegan/ukb_512"
# imageformat="jpg" #원본 이미지 포멧

# # MIV
rhq_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_hq_512"
rlq_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_lq_512" 
new_path="/root/jieunoh/ellen_data/0_cyclegan_input_isecretversion_setnew"
imageformat="jpeg" #원본 이미지 포멧
total_dataset=3


# 0. CV 개수에 따라서 다르게
path_list = []
for num in range(total_dataset):
    numm=num+1
    path_list.append(new_path+"_"+str(numm))

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

# label
csv_test = "/root/jieunoh/ellen_code/ISECRET/test.csv"
csv_train = "/root/jieunoh/ellen_code/ISECRET/train.csv"
csv_val = "/root/jieunoh/ellen_code/ISECRET/val.csv"

df_train = pd.read_csv(csv_train) # dataframe 만들기
train_low = df_train[df_train['label']==0]['name'].values.tolist()
train_high = df_train[df_train['label']==1]['name'].values.tolist()
print("[trian] # of high:%d, low:%d"%(len(train_high),len(train_low)))

df_val = pd.read_csv(csv_val) # dataframe 만들기
val_low = df_val[df_val['label']==0]['name'].values.tolist()
val_high = df_val[df_val['label']==1]['name'].values.tolist()
print("[val] # of high:%d, low:%d"%(len(val_high),len(val_low)))

df_test = pd.read_csv(csv_test) # dataframe 만들기
test_low = df_test[df_test['label']==0]['name'].values.tolist()
test_high = df_test[df_test['label']==1]['name'].values.tolist()
print("[test] # of  high:%d, low:%d"%(len(test_high),len(test_low)))

low_list = train_low+val_low+test_low
high_list = train_high+val_high+test_high

print("\n[A group] ratio of low quality dataset (train : val : test) : (", n_low_train, ": ", n_low_val, ": ",n_low_test ,")")
print("\n[B group] ratio of high quality dataset (train : val) :: (", n_high_train, ": ", n_high_val, ": ",n_high_test, ")")
print("low qulity: ", len(low_list))
print("high qulity: ", len(high_list))
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게

low_name_list = []
for i in low_list:
    low_name_list.append(i.split(".")[0])
high_name_list = []
for i in high_list:
    high_name_list.append(i.split(".")[0])

# 0-3. fold 별로 path------------------------------------------------------------------------------------------------------------


my_low_name_list = []
my_low_list =[]
my_high_name_list = []
my_high_list =[]

for i in os.listdir(rlq_path):
    my_low_name_list.append("_".join(i.split(".")[0].split("_")[0:2]))
    my_low_list.append(i)
for i in os.listdir(rhq_path):
    my_high_name_list.append("_".join(i.split(".")[0].split("_")[0:2]))
    my_high_list.append(i)

# LQ
# 1. real low quality image  불러오기 [A group] -------------------------------
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    random.shuffle(low_name_list)  #  shuffle!! 

    train_count = 0
    val_count= 0
    test_count=0

    for i,img_name in enumerate(tqdm(low_name_list)):
            if i < n_low_test: #test               
                # print("test")
                if img_name in my_low_name_list: 
                    # low path 에서
                    for img in my_low_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rlq_path,img)
                            copy_path = os.path.join(path,"testA",img)
                            shutil.copy(source_path,copy_path)
                            test_count+=1
                else: 
                    for img in my_high_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rhq_path,img)
                            copy_path = os.path.join(path,"testA",img)
                            shutil.copy(source_path,copy_path)
                            test_count+=1
            elif i<(n_low_test+n_low_train):
                # print("train")
                if img_name in my_low_name_list: 
                    # low path 에서
                    for img in my_low_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rlq_path,img)
                            copy_path = os.path.join(path,"trainA",img)
                            shutil.copy(source_path,copy_path)
                            train_count+=1
                else: 
                    for img in my_high_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rhq_path,img)
                            copy_path = os.path.join(path,"trainA",img)
                            shutil.copy(source_path,copy_path)
                            train_count+=1

            else: 
                # print("val")
                if img_name in my_low_name_list: 
                    # low path 에서
                    for img in my_low_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rlq_path,img)
                            copy_path = os.path.join(path,"valA",img)
                            shutil.copy(source_path,copy_path)
                            val_count+=1
                else: 
                    for img in my_high_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rhq_path,img)
                            copy_path = os.path.join(path,"valA",img)
                            shutil.copy(source_path,copy_path)
                            val_count+=1
    print("[now] train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
    print("[real]train/val/test: ", n_low_train, "/", n_low_val, "/",n_low_test)

print("[low quality end]==================================\n\n")



# 2. real hq import image [B group] ------------------------------------------------------
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    random.shuffle(high_name_list)  #  shuffle!! 

    train_count = 0
    val_count= 0
    test_count=0

    for i,img_name in enumerate(tqdm(high_name_list)):
            if i < n_high_test: #test               
                # print("test") 
                if img_name in my_high_name_list: 
                    # high path 에서
                    for img in my_high_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rhq_path,img)
                            copy_path = os.path.join(path,"testB",img)
                            shutil.copy(source_path,copy_path)
                            test_count+=1
                else: 
                    # low path 에서
                    for img in my_high_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rlq_path,img)
                            copy_path = os.path.join(path,"testB",img)
                            shutil.copy(source_path,copy_path)
                            test_count+=1
            elif i<(n_high_test+n_high_train):
                # print("train")
                if img_name in my_high_name_list: 
                    # high path 에서
                    for img in my_high_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rhq_path,img)
                            copy_path = os.path.join(path,"trainB",img)
                            shutil.copy(source_path,copy_path)
                            train_count+=1
                else: 
                    for img in my_high_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rlq_path,img)
                            copy_path = os.path.join(path,"trainB",img)
                            shutil.copy(source_path,copy_path)
                            train_count+=1

            else: 
                # print("test")
                if img_name in my_high_name_list: 
                    # high path 에서
                    for img in my_high_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rhq_path,img)
                            copy_path = os.path.join(path,"valB",img)
                            shutil.copy(source_path,copy_path)
                            val_count+=1
                else: 
                    for img in my_high_list:
                        if img.startswith(img_name):
                            source_path = os.path.join(rlq_path,img)
                            copy_path = os.path.join(path,"valB",img)
                            shutil.copy(source_path,copy_path)
                            val_count+=1
    print("[now] train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
    print("[real]train/val/test: ", n_high_train, "/", n_high_val, "/",n_high_test)

print("[high quality end]==================================\n\n")