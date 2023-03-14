"""
 To make it ressemble fold 3  
 잘되는 fold
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

# image
total_high_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_hq_512"
total_low_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_lq_512"


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
n_low_val = 4558  # test
n_low_test = 178

# high_B
n_high_train = 7482
n_high_val = 8471  #test
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



# 0-3. fold 별로 path------------------------------------------------------------------------------------------------------------
# fold 별로 좋은 것 - fold 2,3
testA_fold2 = "/root/jieunoh/ellen_data/0_cyclegan_input/cginput_eyeq_total_2/testA"
testB_fold2 = "/root/jieunoh/ellen_data/0_cyclegan_input/cginput_eyeq_total_2/testB"
good2_high_list= os.listdir(testB_fold2)

testA_fold3= "/root/jieunoh/ellen_data/0_cyclegan_input/cginput_eyeq_total_3/testA"
testB_fold3= "/root/jieunoh/ellen_data/0_cyclegan_input/cginput_eyeq_total_3/testB"
good3_high_list= os.listdir(testB_fold3)
# fold 안좋은 것 - fold 1
testA_fold1= "/root/jieunoh/ellen_data/0_cyclegan_input/cginput_eyeq_total_1/testA"
testB_fold1= "/root/jieunoh/ellen_data/0_cyclegan_input/cginput_eyeq_total_1/testB"



bad_lq_list= os.listdir(testA_fold1)

# LQ
# 1. real low quality image  불러오기 [A group] --------------------------------------------------------------

print("------------------------------------------------------")
print("low quality 구성")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    train_count = 0
    val_count= 0
    test_count=0
    val_list =[]

    print("fold2")
    for img in tqdm(os.listdir(testA_fold2)): 
        name = img.split(".")[0].split("_")[0:2]
        if imageformat in img:
            source_path = os.path.join(testA_fold2,img)
            copy_path = os.path.join(path,"valA",img)
            shutil.copy(source_path,copy_path)
            val_count+=1
    print("fold3")
    for img in tqdm(os.listdir(testA_fold3)):
        name = img.split(".")[0].split("_")[0:2]
        if imageformat in img :
            source_path = os.path.join(testA_fold3,img)
            copy_path = os.path.join(path,"valA",img)
            shutil.copy(source_path,copy_path)
            val_count+=1

    print("fold1")
    random.shuffle(bad_lq_list)
    for img in tqdm(bad_lq_list): 
        if imageformat in img: 
            if val_count <n_low_val: # val 마무리
                source_path = os.path.join(testA_fold1,img)
                copy_path = os.path.join(path,"valA",img)
                shutil.copy(source_path,copy_path)
                val_count+=1

            elif test_count<n_low_test: # test
                source_path = os.path.join(testA_fold1,img)
                copy_path = os.path.join(path,"testA",img)
                shutil.copy(source_path,copy_path)
                test_count+=1
            else: # train
                source_path = os.path.join(testA_fold1,img)
                copy_path = os.path.join(path,"trainA",img)
                shutil.copy(source_path,copy_path)
                train_count+=1
    print("train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
print("[low quality end]==================================\n\n")



# 2. real hq import image [B group] ------------------------------------------------------
print("------------------------------------------------------")
print("high quality 구성")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    train_count = 0
    val_count= 0
    test_count=0
    
    if index == 0:
        # print("fold2")
        for img in tqdm(os.listdir(testB_fold2)): 
            if imageformat in img:# val
                source_path = os.path.join(testB_fold2,img)
                copy_path = os.path.join(path,"valB",img)
                shutil.copy(source_path,copy_path)
                val_count+=1

        random.shuffle(good3_high_list)
        for img in tqdm(good3_high_list):
            if imageformat in img:
                if val_count<n_high_val:
                    source_path = os.path.join(testB_fold3,img)
                    copy_path = os.path.join(path,"valB",img)
                    shutil.copy(source_path,copy_path)
                    val_count+=1
                elif test_count < n_high_test: # test
                    source_path = os.path.join(testB_fold3,img)
                    copy_path = os.path.join(path,"testB",img)
                    shutil.copy(source_path,copy_path)
                    test_count+=1
                else: # train
                    source_path = os.path.join(testB_fold3,img)
                    copy_path = os.path.join(path,"trainB",img)
                    shutil.copy(source_path,copy_path)
                    train_count += 1

        for img in tqdm(os.listdir(testB_fold1)):
            if imageformat in img:
                source_path = os.path.join(testB_fold1,img)
                copy_path = os.path.join(path,"trainB",img)
                shutil.copy(source_path,copy_path)
                train_count += 1

    else:
        #index 2,3
        # fold 3
        for img in tqdm(os.listdir(testB_fold3)): 
            if imageformat in img:# val
                source_path = os.path.join(testB_fold3,img)
                copy_path = os.path.join(path,"valB",img)
                shutil.copy(source_path,copy_path)
                val_count+=1

        random.shuffle(good2_high_list)
        for img in tqdm(good2_high_list):
            if imageformat in img:
                if val_count<n_high_val:
                    source_path = os.path.join(testB_fold2,img)
                    copy_path = os.path.join(path,"valB",img)
                    shutil.copy(source_path,copy_path)
                    val_count+=1
                elif test_count < n_high_test: # test
                    source_path = os.path.join(testB_fold2,img)
                    copy_path = os.path.join(path,"testB",img)
                    shutil.copy(source_path,copy_path)
                    test_count+=1
                else: # train
                    source_path = os.path.join(testB_fold2,img)
                    copy_path = os.path.join(path,"trainB",img)
                    shutil.copy(source_path,copy_path)
                    train_count += 1

        for img in tqdm(os.listdir(testB_fold1)):
            if imageformat in img:
                source_path = os.path.join(testB_fold1,img)
                copy_path = os.path.join(path,"trainB",img)
                shutil.copy(source_path,copy_path)
                train_count += 1

    print("train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
print("[high quality end]==================================\n\n")
