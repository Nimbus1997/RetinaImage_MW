import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import shutil
import pdb
"""
Ellen Made
date: 2023.02.14 (server: miv4)

> input_eyeq_total_spilt_new 에 A에는 lq가 B에는 hq만 잘 들어갔는지 확인하기 위해서
>  "/root/jieunoh/ellen_data/input_eyeq_total_spilt_new/valA" 에 high quality (label =0) 이 있는지 확인 하기 위해서 
>  (testA)에는 high quality가 들어가 있음 -> 근데 h라는 걸 안써놓음

[label]
0: high (B)
1: usable (A)
2: reject (없음)

"""

# 0. setting 
# source ################
# label
csv_test = "/root/jieunoh/ellen_code/eyeQ_ellen/data/Label_EyeQ_test.csv"
csv_train = "/root/jieunoh/ellen_code/eyeQ_ellen/data/Label_EyeQ_train.csv"

# image
image_path = "/root/jieunoh/ellen_data/input_eyeq_total_spilt_new_5/valA"

# the label you want to check
desired_label = 1
# [label]
# 0: high (B)
# 1: usable (A)
# 2: reject (없음)


print("image_path: ",image_path)
print("desired_label (0:high B, 1: usable A, 2: reject): ",desired_label)
input("위의 값 확인후 enter 눌러서 진행 >>>") 


###################################################################################
# 1. 원하는 label인 image name list 만들기
df_test = pd.read_csv(csv_test) # dataframe 만들기
testimage_list = df_test[df_test['quality']==desired_label]['image'].to_list() # quality 가 2 인 image 이름가져오기
df_train = pd.read_csv(csv_train) # dataframe 만들기
trainimage_list = df_train[df_train['quality']==desired_label]['image'].to_list() # quality 가 2 인 image 이름가져오기

desired_image = testimage_list+trainimage_list
# print(desired_image)

# 2. 해당 image path에서 그 label이 아닌 것 찾기
count =0
for imagename in tqdm(os.listdir(image_path)):
    name = "_".join(imagename.split("_")[:2])+".jpeg"
    if name not in desired_image: 
        count+=1
print("WRONG", count)
    





