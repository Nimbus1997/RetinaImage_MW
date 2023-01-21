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
csv_path_list = ["/root/jieunoh/ellen_code/eyeQ_ellen/data/Label_EyeQ_test.csv", "/root/jieunoh/ellen_code/eyeQ_ellen/data/Label_EyeQ_train.csv"]
csv_test = "/root/jieunoh/ellen_code/eyeQ_ellen/data/Label_EyeQ_test.csv"
csv_train = "/root/jieunoh/ellen_code/eyeQ_ellen/data/Label_EyeQ_train.csv"
# image
test_path = "/root/jieunoh/ellen_data/EyeQ/test_ori"
train_path = "/root/jieunoh/ellen_data/EyeQ/train_ori"

# new - generation ################
new_path = "/root/jieunoh/ellen_data/EyeQ/reject"
if not os.path.isdir(new_path):
    os.makedirs(new_path)


print("csv_test",csv_test)
print("csv_train",csv_train)
print("test_path",test_path)
print("train_path",train_path)
print("new_path",new_path)
input("위의 값 확인후 enter 눌러서 진행 >>>") 


###################################################################################
# 1. [test] quality 가 2 인 image 이름가져오기 
df_test = pd.read_csv(csv_test) # dataframe 만들기
reject_image_testlist = df_test[df_test['quality']==2]['image'] # quality 가 2 인 image 이름가져오기 

# 2. [test] reject image를 new path에 복사하기
count =0
for imagename in tqdm(reject_image_testlist):
    source_path = test_path+"/"+imagename
    copy_path =new_path+"/"+imagename
    shutil.copyfile(source_path, copy_path)
    count+=1
print("in test.csv the # of reject : ", count)
    

# 3. [train] quality 가 2 인 image 이름가져오기 
df_train = pd.read_csv(csv_train) # dataframe 만들기
reject_image_trainlist = df_train[df_train['quality']==2]['image'] # quality 가 2 인 image 이름가져오기 

# 4. [train] reject image를 new path에 복사하기
count =0
for imagename in tqdm(reject_image_trainlist):
    source_path = train_path+"/"+imagename
    copy_path =new_path+"/"+imagename
    shutil.copyfile(source_path, copy_path)
    count+=1
print("in train.csv the # of reject : ", count)





