"""
 [input_20220623_#_n1000]를 for Cross Validation을 위해서 
 PART 1 []
 1. hq와 lq모우기-> *_hq , *_lq 폴더에 생성 0001부터 - 1000까지
 PART 2 [v]
 2. 5 cross val 만들기 
    (1) RetinaImage_MW/data_transform/input_cycleGAN/For_CV_spliting_data.py based
    (2) 4개 더 만들기
 
 date: 2023.01.18
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
# rhq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220623_512_n1000_hq"
# rlq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220623_512_n1000_lq" 
# path2="input_20221211_2_512_n1000"
# path3="input_20221211_3_512_n1000"
# gen_path2 = "/home/guest1/ellen_data/UKB_quality_data2_combined/"+path2
# gen_path3 = "/home/guest1/ellen_data/UKB_quality_data2_combined/"+path3

# imageformat="jpg" #원본 이미지 포멧

# MIV
rhq_path = "/root/jieunoh/ellen_data/isecret_input_eyeq_total_hq"
rlq_path = "/root/jieunoh/ellen_data/isecret_input_eyeq_total_lq" 
new_path="/root/jieunoh/ellen_data/isecret_input_eyeq_total_new"

imageformat="jpeg" #원본 이미지 포멧
num_how_much_more_dataset=4

# 0. CV 개수에 따라서 다르게
path_list = []
for num in range(num_how_much_more_dataset):
    numm=num+2
    path_list.append(new_path+"_"+str(numm))


trainlR, vallR, testlR = 6, 2, 2  # low quality data의 train : val: test 비율 쓰기
trainhR, valhR, testhR  = 6, 2, 2 # high qualtiy data의 train : val: test 비율 쓰기
biggerthanthis = 10000  # 10000보단 작을 것으로 예상

# 00. seting 된 parameter 확인  ----------------------------------------------------------------------------------------- 


print("path check")
print("rlq_path: ", rlq_path)
print("rhq_path: ", rhq_path)
print("num_how_much_more_dataset:", num_how_much_more_dataset)
for path in path_list:
    print("gen_path:", path)

print("원본 이미지 포멧: ", imageformat)

print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게

group1 = ["train","val","test"]  # outer directory
group2 = ["crop_good","crop_usable"]  # inner directories 

# 0-1. path없으면 path생성 ------------------------------------------------------------------------------------------
for gen_path in path_list:
    print("--------------------------")
    print(gen_path)
    for g1 in group1:
        print(g1)
        for g2 in group2:
            if not os.path.isdir(gen_path+"/eyeq/"+g1+"/"+g2):
                os.makedirs(gen_path+"/eyeq/"+g1+"/"+g2)

# 0-2. dataset 나누기 -train val test 개수------------------------------------------------------------------------------------------------------------

# low qualtiy -> A group
ltotal_dataset = len(os.listdir(rlq_path))

ltrainN = int(float(ltotal_dataset)/10.*float(trainlR))
lvalN = int(float(ltotal_dataset)/10.*float(valhR))
ltestN = ltotal_dataset - ltrainN-lvalN

ldigit = len(str(ltotal_dataset)) # 자릿수
print("\n-------------------------------")
print("ldigit: ", ldigit)

# high qualtiy -> B group & A group in test
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
# 1. real low quality image  불러오기 [crop_usable -group2[1]] -------------------------------
low_quality_block_size=ltotal_dataset/5
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    test_start = low_quality_block_size*index

    if index==0:
        val_start=low_quality_block_size*(4)
    else:
        val_start=low_quality_block_size*(index-1)

    i=0
    test_count=0
    val_count=0
    train_count=0
    print(str(test_start),"/", str(test_start+low_quality_block_size))
    print(str(val_start),"/", str(val_start+low_quality_block_size))

    for rlq in tqdm(sorted(os.listdir(rlq_path))):
        if imageformat in rlq:
            source_path=rlq_path+'/'+rlq

        #2. real lq image 저장하기 ------------------
            # test
            if (i>=test_start) and (i<(test_start+low_quality_block_size)): 
                copy_path=path+"/eyeq/test/"+group2[1]
                shutil.copy(source_path,copy_path)
                test_count+=1
            # val
            elif (i>=val_start) and (i<(val_start+low_quality_block_size)):
                copy_path =path+"/eyeq/val/"+group2[1]         
                shutil.copy(source_path,copy_path)
                val_count += 1
            # train
            else:
                copy_path =path+"/eyeq/train/"+group2[1]         
                shutil.copy(source_path,copy_path)
                train_count += 1
            i += 1
    print("train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
print("[low quality end]==================================\n\n")



# 2. real hq import image [crop_good group2[0]] ------------------------------------------------------
high_quality_block_size=htotal_dataset/5
print("------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    test_start = high_quality_block_size*index

    if index==0:
        val_start=high_quality_block_size*(4)
    else:
        val_start=high_quality_block_size*(index-1)

    i=0
    test_count=0
    val_count=0
    train_count=0
    print(str(test_start),"/", str(test_start+high_quality_block_size))
    print(str(val_start),"/", str(val_start+high_quality_block_size))

    for rhq in tqdm(sorted(os.listdir(rhq_path))):
        if imageformat in rhq:
            source_path=rhq_path+'/'+rhq

        #2. real lq image 저장하기 ------------------
            # test
            if (i>=test_start) and (i<(test_start+high_quality_block_size)): 
                copy_path=path+"/eyeq/test/"+group2[0]
                shutil.copy(source_path,copy_path)
                test_count+=1
            # val
            elif (i>=val_start) and (i<(val_start+high_quality_block_size)):
                copy_path = path+"/eyeq/val/"+group2[0]  
                shutil.copy(source_path,copy_path)
                val_count += 1
            # train
            else:
                copy_path = path+"/eyeq/train/"+group2[0]
                shutil.copy(source_path,copy_path)
                train_count += 1
            
            i += 1
    print("train/val/test: ", str(train_count),"/",str(val_count),"/",str(test_count))
print("[high quality end]==================================\n\n")
