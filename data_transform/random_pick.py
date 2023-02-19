import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import shutil
import random

"""
Ellen Made
date: 2023.02.19 (server: miv4)
> random pick 1000 reject images from 5535 images.

"""

# 0. setting 
ori_path = "/root/jieunoh/ellen_data/reject_crop_512"
new_path = "/root/jieunoh/ellen_data/reject_crop_512_n1000"
how_many=1000

if not os.path.isdir(new_path):
    os.makedirs(new_path)

print("ori_path: ",ori_path)
print("new_path: ",new_path)
input("위의 값 확인후 enter 눌러서 진행 >>>") 


###################################################################################
image_list = os.listdir(ori_path)  # image name list
random_choice_list=random.sample(image_list,how_many) # ranom # samples of the images

count =0
for sample_image in tqdm(random_choice_list):
    source_path = ori_path+"/"+sample_image
    copy_path =new_path+"/"+sample_image
    shutil.copyfile(source_path, copy_path)
    count+=1
print("how many images are sampled : ", count)





