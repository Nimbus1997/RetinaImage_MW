"""
 [input_20220623_#_n1000]를 for Cross Validation을 위해서 
 PART 1 []
 1. hq와 lq, degrade 모으기-> *_hq , *_lq 폴더에 생성
 PART 2 [v]
 2. 3 cross val 만들기 
    (1) RetinaImage_MW/data_transform/input_cycleGAN/For_CV_spliting_data_onlyValTrain.py based
    (2) 3 개의 cross validation dataset 만들기
    (3) 각 3개의 cross validaion dataset에  
        (i) 모든 데이터 사용
        (ii) hq에 들어간 것을 degrade에서 찾아서 같이 구성
        (iii) train : val = 2:1 로 구성-> test와 val은 같게 구성
 
 date: 2023.01.27
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
degrade_path= '/root/jieunoh/ellen_data/isecret_eyeq_total_degraded'
new_path="/root/jieunoh/ellen_data/0_isceret_input/isinput_eyeq_total"

imageformat="jpg" #원본 이미지 포멧

# MIV
# rhq_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_hq"
# rlq_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_lq" 
# degrade_path= '/root/jieunoh/ellen_data/isecret_eyeq_total_degraded'
# new_path="/root/jieunoh/ellen_data/0_isceret_input/isinput_eyeq_total"

# imageformat="jpeg" #원본 이미지 포멧


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
print("degrade_path: ", degrade_path)

print("how_much_dataset:", total_dataset)
for path in path_list:
    print("gen_path:", path)

print("원본 이미지 포멧: ", imageformat)

print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게

group1 = ["train","val","test"]  # outer directory
group2 = ["crop_good","crop_usable","degrade_good"]  # inner directories 

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
# 1. real low quality image  불러오기 [crop_usable -group2[1]] -------------------------------
low_quality_block_size=ltotal_dataset/total_dataset
print("LQ------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    val_start = low_quality_block_size*index

    i=0
    val_count=0
    train_count=0

    print(str(val_start),"/", str(val_start+low_quality_block_size))

    for rlq in tqdm(sorted(os.listdir(rlq_path))):
        if imageformat in rlq:
            source_path=rlq_path+'/'+rlq

        #2. real lq image 저장하기 ------------------
            # val = test
            if (i>=val_start) and (i<(val_start+low_quality_block_size)): 
                copy_path_val =path+"/eyeq/val/"+group2[1]         
                shutil.copy(source_path,copy_path_val)

                copy_path_test=path+"/eyeq/test/"+group2[1]
                shutil.copy(source_path,copy_path_test)
                val_count+=1
        
            # train
            else:
                copy_path =path+"/eyeq/train/"+group2[1]         
                shutil.copy(source_path,copy_path)
                train_count += 1
            i += 1
    print("train/val=test: ", str(train_count),"/",str(val_count))
print("[low quality end]==================================\n\n")



# 2. real hq import image [crop_good group2[0]] & degraded image ------------------------------------------------------
degraded_set=['001', '100', '010', '110', '101', '011', '111']

high_quality_block_size=htotal_dataset/total_dataset
print("HQ, degrade------------------------------------------------------")
for index, path in enumerate(path_list):
    print("--[",path,"]-----------------------------------------")
    val_start = high_quality_block_size*index

    i=0
    val_count=0
    train_count=0

    print(str(val_start),"/", str(val_start+high_quality_block_size))

    for rhq in tqdm(sorted(os.listdir(rhq_path))):
        if imageformat in rhq:
            source_path=rhq_path+'/'+rhq
            name=rhq.split(".")[0]

        #2. real lq image 저장하기 ------------------
            # val =test
            if (i>=val_start) and (i<(val_start+high_quality_block_size)): 
                # val = test # crop_good
                copy_path_val = path+"/eyeq/val/"+group2[0]  
                shutil.copy(source_path,copy_path_val)
                copy_path_test=path+"/eyeq/test/"+group2[0]
                shutil.copy(source_path,copy_path_test)
                # degraded
                for degrade in degraded_set:
                    source_degrade_path = degrade_path+'/'+name+'_'+degrade+".jpeg"
                    copy_degrade_path_val = path+"/eyeq/val/"+group2[2]
                    shutil.copy(source_degrade_path,copy_degrade_path_val)
                    copy_degrade_path_test = path+"/eyeq/test/"+group2[2]
                    shutil.copy(source_degrade_path,copy_degrade_path_test)

                val_count+=1
            
            # train
            else:
                copy_path = path+"/eyeq/train/"+group2[0]
                shutil.copy(source_path,copy_path)
                for degrade in degraded_set:
                    source_degrade_path = degrade_path+'/'+name+'_'+degrade+".jpeg"
                    copy_degrade_path = path+"/eyeq/train/"+group2[2]
                    shutil.copy(source_degrade_path,copy_degrade_path)
                train_count += 1
            
            i += 1
    print("train/val=test: ", str(train_count),"/",str(val_count))
print("[high quality end]==================================\n\n")
