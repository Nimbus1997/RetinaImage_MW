import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import shutil

"""
Ellen Made
date: 2023.01.20 (server: miv2)
edit:

> 참고: posco AI 실습자료 > '00_01Docker Python Pandas lab.ppt'
> to split the eyeq image source according by the label made by eyeQ "https://github.com/HzFu/EyeQ/tree/master/data"

0. 이미 good[0]과 usable[1]은 각각 hq, lq로 나눠놓음
1. reject[2]만 새로 만들기
2. label 위치: eyeQ_ellen/data/Label_EyeQ_test.csv"
3. image 위치: miv2 server jieunoh/ellen_data/EyeQ/train_ori or test_ori 

"""

# 0. setting 
# source ################
# label
csv_test = "/root/jieunoh/ellen_code/ISECRET/test.csv"
csv_train = "/root/jieunoh/ellen_code/ISECRET/train.csv"
csv_val = "/root/jieunoh/ellen_code/ISECRET/val.csv"
# image
high_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_hq_512"
low_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_lq_512"
degrade_path ="/root/jieunoh/ellen_data/isecret_eyeq_total_degrade1"


# new - generation ################
gen_path = "/root/jieunoh/ellen_data/0_cyclegan_input_isecretversion"
if not os.path.isdir(gen_path):
    os.makedirs(gen_path)

print("csv_train",csv_train)
print("csv_val",csv_val)
print("csv_test",csv_test)
print("high_path",high_path)
print("low_path",low_path)
print("degrade_path",degrade_path)

print("gen_path",gen_path)
input("위의 값 확인후 enter 눌러서 진행 >>>") 


###################################################################################
# 1.train / val /test image name list가져오기 
df_train = pd.read_csv(csv_train) # dataframe 만들기
train_low = df_train[df_train['label']==0]['name']
train_high = df_train[df_train['label']==1]['name']
print("[trian] # of high:%d, low:%d"%(len(train_high),len(train_low)))

df_val = pd.read_csv(csv_val) # dataframe 만들기
val_low = df_val[df_val['label']==0]['name']
val_high = df_val[df_val['label']==1]['name']
print("[val] # of high:%d, low:%d"%(len(val_high),len(val_low)))

df_test = pd.read_csv(csv_test) # dataframe 만들기
test_low = df_test[df_test['label']==0]['name']
test_high = df_test[df_test['label']==1]['name']
print("[test] # of  high:%d, low:%d"%(len(test_high),len(test_low)))


###################################################################################
# 2. cycle gan input 구성 
# 2-1. 폴더 만들기
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
    

# 2-2. 데이터 옮기기 (train)
count =0 
for trian_low_img in tqdm(train_low):
    name="_".join(trian_low_img.split(".")[0].split("_")[0:2])
    for low in os.listdir(low_path):
        if low.startswith(name):
            source_path =  os.path.join(low_path,low)
            copy_path = os.path.join(gen_path,"trainA",low)
            shutil.copy(source_path,copy_path)
            count +=1
if len(train_low) == count :
    print("[trainA] matched !")

count =0 
for trian_high_img in tqdm(train_high):
    name="_".join(trian_high_img.split(".")[0].split("_")[0:2])
    for high in os.listdir(high_path):
        if high.startswith(name):
            source_path =  os.path.join(high_path,high)
            copy_path = os.path.join(gen_path,"trainB",high)
            shutil.copy(source_path,copy_path)
            count +=1
if len(train_high) == count :
    print("[trainB] matched !")



# 2-2. 데이터 옮기기 (val)

count =0 
for val_low_img in tqdm(val_low):
    name="_".join(val_low_img.split(".")[0].split("_")[0:2])
    for low in os.listdir(low_path):
        if low.startswith(name):
            source_path =  os.path.join(low_path,low)
            copy_path = os.path.join(gen_path,"valA",low)
            shutil.copy(source_path,copy_path)
            count +=1
if len(val_low) == count :
    print("[valA] matched !")

count =0 
for val_high_img in tqdm(val_high):
    name="_".join(val_high_img.split(".")[0].split("_")[0:2])
    for high in os.listdir(high_path):
        if high.startswith(name):
            source_path =  os.path.join(high_path,high)
            copy_path = os.path.join(gen_path,"valB",high)
            shutil.copy(source_path,copy_path)
            count +=1
if len(val_high) == count :
    print("[valB] matched !")


# 2-2. 데이터 옮기기 (test)


count =0 
for test_low_img in tqdm(test_low):
    name="_".join(test_low_img.split(".")[0].split("_")[0:2])
    for low in os.listdir(low_path):
        if low.startswith(name):
            source_path =  os.path.join(low_path,low)
            copy_path = os.path.join(gen_path,"testA",low)
            shutil.copy(source_path,copy_path)
            count +=1
if len(test_low) == count :
    print("[testA] matched !")

count =0 
for test_high_img in tqdm(test_high):
    name="_".join(test_high_img.split(".")[0].split("_")[0:2])
    for high in os.listdir(high_path):
        if high.startswith(name):
            source_path =  os.path.join(high_path,high)
            copy_path = os.path.join(gen_path,"testB",high)
            shutil.copy(source_path,copy_path)
            count +=1
if len(test_high) == count :
    print("[testB] matched !")

