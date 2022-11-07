######################################
# 이미지 파일 이름 순서대로 바꾸기 
######################################
# outdoor data가 이름이 마음대로 되어 있음. 
# outdoor_rename 폴더로 다시 이름 바꾸기 
# 001.png순으로
#
# 2022.07.10
# Ellen Jieun Oh 
# jieunoh@postech.ac.kr

######################################

import os
import cv2 as cv
import numpy as np 
from PIL import Image


ori_path = "/home/guest1/ellen_data/outdoor"
gen_path ="/home/guest1/ellen_data/outdoor_rename"

folders = ['trainA','trainB', 'valA', 'valB', 'testA','testB']
ldigit = 3

# 폴더 없으면, 새로 만들기
if not os.path.isdir(gen_path):
    for f in folders:
        os.makedirs(gen_path+"/"+f)
        print(f+ " generated!")

for f in folders:
    i =0 
    for img in sorted(os.listdir(ori_path+"/"+f)):
        imgg = Image.open(ori_path+"/"+f+"/"+img)
        imgg.save(gen_path+"/"+f+"/"+str(i).zfill(ldigit)+".jpg")
        i +=1
