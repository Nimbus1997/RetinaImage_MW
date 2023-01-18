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

# # Mediwhale
# original_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220623_512_n1000"
# lq_path = "/".join(original_path.split("/")[:-1]+[(original_path.split("/")[-1]+"_lq")])
# hq_path = "/".join(original_path.split("/")[:-1]+[(original_path.split("/")[-1]+"_hq")])
# file_sy ="jpg"

# MIV
original_path="/root/jieunoh/ellen_data/input_eyeq_total_spilt_new"
lq_path = "/".join(original_path.split("/")[:-1]+[(original_path.split("/")[-1]+"_lq")])
hq_path = "/".join(original_path.split("/")[:-1]+[(original_path.split("/")[-1]+"_hq")])
file_sy ="jpeg"


l= 0
h= 0 

ldigit= 4 # 4자리수로 맞추기
changename=False
# changename=True


print("original_path:", original_path)
print("lq_path:", lq_path)
print("hq_path:", hq_path)
print("file 확장자: ",file_sy)
input("위의 값 확인후 enter 눌러서 진행 >>>")

if not os.path.isdir(lq_path):
    os.makedirs(lq_path)
    print(lq_path+"generated!")
if not os.path.isdir(hq_path):
    os.makedirs(hq_path)
    print(hq_path+"generated!")

groupA = ["trainA","valA","testA"]  # low quality - testA에 hq도 있음
groupB = ["trainB","valB","testB"]  # high quality 

if changename:
    print("low (and high in test) quality images extracting - - -")
    for group in groupA:
        for image in tqdm(sorted(os.listdir(original_path+"/"+group))):
            if image[0] == 'h':
                source_path=original_path+"/"+group+"/"+image
                copy_path=hq_path+"/"+str(h).zfill(ldigit)+"."+file_sy
                shutil.copy(source_path,copy_path)
                h+=1
            else:
                source_path=original_path+"/"+group+"/"+image
                copy_path=lq_path+"/"+str(l).zfill(ldigit)+"."+file_sy
                shutil.copy(source_path,copy_path)
                l=l+1
    print("high quality images extracting - - -")
    for group in groupB:
        for image in tqdm(sorted(os.listdir(original_path+"/"+group))):
            source_path=original_path+"/"+group+"/"+image
            copy_path=hq_path+"/"+str(h).zfill(ldigit)+"."+file_sy
            shutil.copy(source_path,copy_path)

            h=h+1   

else:
    print("not changing the name")
    print("low (and high in test) quality images extracting - - -")
    for group in groupA:
        for image in tqdm(sorted(os.listdir(original_path+"/"+group))):
            if image[0] == 'h':
                source_path=original_path+"/"+group+"/"+image
                copy_path=hq_path+"/"+image
                shutil.copy(source_path,copy_path)
                h+=1
            else:
                source_path=original_path+"/"+group+"/"+image
                copy_path=lq_path+"/"+image
                shutil.copy(source_path,copy_path)
                l=l+1
    print("high quality images extracting - - -")
    for group in groupB:
        for image in tqdm(sorted(os.listdir(original_path+"/"+group))):
            source_path=original_path+"/"+group+"/"+image
            copy_path=hq_path+"/"+image
            shutil.copy(source_path,copy_path)

            h=h+1   



print("# of hq: ", h,", # of lq: ", l)