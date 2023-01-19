"""
 [input_20220623_#_n1000]를 for Cross Validation을 위해서 
 PART 1
 1. hq와 lq모우기-> *_hq , *_lq 폴더에 생성 0001부터 - 1000까지
 2. 3 cross val 만들기 
    (1) 원래처럼 train- val -test 순
    (2) test-train-val 순
    (3) val-test-train 순
 
 date: 2022.12.10
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
import pdb


# MIV
original_path="/root/jieunoh/ellen_data/input_eyeq_isecret_total_spilt_new/eyeq"
gen_path ="isecret_input_eyeq_total"
lq_path = "/".join(original_path.split("/")[:-2]+[gen_path+"_lq"])
hq_path = "/".join(original_path.split("/")[:-2]+[gen_path+"_hq"])
degraded_path ="/".join(original_path.split("/")[:-2]+[gen_path+"_degraded"])
file_sy ="jpeg"


l= 0
h= 0 



print("original_path:", original_path)
print("lq_path:", lq_path)
print("hq_path:", hq_path)
print("degraded_path:", degraded_path)
print("file 확장자: ",file_sy)
input("위의 값 확인후 enter 눌러서 진행 >>>")

if not os.path.isdir(lq_path):
    os.makedirs(lq_path)
    print(lq_path+"generated!")
if not os.path.isdir(hq_path):
    os.makedirs(hq_path)
    print(hq_path+"generated!")
if not os.path.isdir(degraded_path):
    os.makedirs(degraded_path)
    print(degraded_path+"generated!")

group1 = ["train","val","test"]  # outer directory
group2 = ["crop_good","crop_usable","degrade_good"]  # inner directories 

print("not changing the name")
print("low (and high in test) quality images extracting - - -")
total_high_count=0
total_low_count=0
total_degrade_count=0

for group11 in group1:
    for i, group22 in enumerate(group2):
        count =0
        if i==0:
            # crop_good
            print("high quality images extracting - - -")
            for image in tqdm(sorted(os.listdir(original_path+"/"+group11+"/"+group22))):
                source_path=original_path+"/"+group11+"/"+group22+"/"+image
                copy_path=hq_path+"/"+image
                shutil.copy(source_path,copy_path)
                count+=1
            total_high_count+=count
            print("# of hq data:", count)

        elif i==1:
            # crop_usable
            print("low quality images extracting - - -")
            for image in tqdm(sorted(os.listdir(original_path+"/"+group11+"/"+group22))):
                source_path=original_path+"/"+group11+"/"+group22+"/"+image
                copy_path=lq_path+"/"+image
                shutil.copy(source_path,copy_path)
                count+=1
            total_low_count+=count
            print("# of lq data:", count)

        else:
            # degrade_good
            print("degrade_good images extracting - - -")
            for image in tqdm(sorted(os.listdir(original_path+"/"+group11+"/"+group22))):
                source_path=original_path+"/"+group11+"/"+group22+"/"+image
                copy_path=degraded_path+"/"+image
                shutil.copy(source_path,copy_path)
                count+=1
            total_degrade_count+=count
            print("# of degraded data:", count)

print("-----------------------------------------------")
print("total_high_count: ", total_high_count)
print("total_low_count: ", total_low_count)
print("total_degrade_count: ", total_degrade_count)
