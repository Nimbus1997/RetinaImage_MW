# ------------------------------------------------
# read me
# > for input data
# > shaprness.py와 illumination.py의 함수를 이용해서 최종 quality score를 구해서 excel에 작성
# > excel의 column: [1] folder(ex.trainA), 
#                   [2] High quality or Low quality
#                   [3] ImageName,
#                   [4] sharpness, 
#                   [5] illumination, 
#                   [6] quality score, 
#                   [7] category(1: verygood, 2:good, 3: acceptable, 4 not acceptable)
# > excel name: AQE_data폴더이름_date
# AQE: by H.Bartling ‘Automated Quality Evaluation of Digital Fundus Photography’ 2009 Acta Ophthalmologica (impact factor: 3.761, # cite: 68)
# > 2022.06.23 jieunoh@postech.ac.kr
# ------------------------------------------------
from requests import patch
from illumination import illumination
from sharpness import sharpness
import numpy as np
import os
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd


# 0. 돌릴때 마다 설정 ---------------------------
image_size = 128
patch_size = 32 
path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220623_128_n1000/"
date = '0623'


# 0. 설정 ---------------------------
excel_path ="/home/guest1/ellen_code/RetinaImage_MW/image_evaluation_AQE/AQE_result/"# no need to change this
excel_name = "AQE_"+path.split("/")[-2]+"_" +date+".csv"

foldersA = ['trainA', 'valA'] # testA는 따로 (l & h 섞여있어서)
foldersB = ['trainB','valB', 'testB']

num = image_size/patch_size
print("-------------------------------")
print("image size: ",image_size)
print("patch_size: ",patch_size)
print("orignal_data: ",path.split("/")[-2])
print("output_name: ",excel_name)
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게

# 0. def category ---------------------------
def category(quality_score:int)->int:
    if quality_score>10:
        #very good
        return 1
    elif quality_score>6:
        #good
        return 2
    elif quality_score>2:
        #acceptable
        return 3
    else:
        #un acceptable
        return 4


# 1. excel 기본 설정 ---------------------------
df = pd.DataFrame(columns=['folder','quality','image_name','sharpness','illumination','quality_score','category'])
count =0

# 2. main _A  
qualityi='l'
for folder in foldersA:
    for image in sorted(os.listdir(path+folder)):
        img =cv.imread(path+folder+"/"+image,0)
        sharpnessi=sharpness(img,num)
        illuminationi=illumination(img,num)
        quality_scorei=sharpnessi+illuminationi
        categoryi=category(quality_scorei)

        row =[folder,qualityi, image, sharpnessi, illuminationi, quality_scorei, categoryi]
        df.loc[count] = row
        count+=1
    print(">>>>END: "+folder)

# 3. main _testA ---------------------------
for image in sorted(os.listdir(path+"testA")):
    img =cv.imread(path+"testA/"+image,0)
    qualityi='l'
    sharpnessi=sharpness(img,num)
    illuminationi=illumination(img,num)
    quality_scorei=sharpnessi+illuminationi
    categoryi=category(quality_scorei)
    if "h" in image:
        qualityi='h'
    row =[folder,qualityi, image, sharpnessi, illuminationi, quality_scorei, categoryi]
    df.loc[count] = row
    count+=1
print(">>>>END: testA")


# 4. main _ B  ---------------------------
qualityi='h'
for folder in foldersB:
    for image in sorted(os.listdir(path+folder)):
        img =cv.imread(path+folder+"/"+image,0)
        sharpnessi=sharpness(img,num)
        illuminationi=illumination(img,num)
        quality_scorei=sharpnessi+illuminationi
        categoryi=category(quality_scorei)

        row =[folder,qualityi, image, sharpnessi, illuminationi, quality_scorei, categoryi]
        df.loc[count] = row
        count+=1
    print(">>>>END: "+folder)
    


# 5. make it to csv  ---------------------------
df.to_csv(excel_path +excel_name)
print("-----------------------------")
print("cvs saved!!! : "+excel_name)
print("path: "+excel_path+excel_name)
print("-----------------------------")
