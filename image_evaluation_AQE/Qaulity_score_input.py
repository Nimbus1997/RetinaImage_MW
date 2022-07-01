# ------------------------------------------------
# read me
# > for input data
# > shaprness.py와 illumination.py의 함수를 이용해서 최종 quality score를 구해서 excel에 작성
# > excel의 column: [1] folder(ex.trainA), 
#                   [2] High quality or Low quality
#                   [3] ImageName,
#                   [4] sharpness, 
#                   [5] illumination, 
#                   [6] quality_score, 
#                   [7] category(1: verygood, 2:good, 3: acceptable, 4 not acceptable)
# > excel name: AQE_data폴더이름_date
# AQE: by H.Bartling ‘Automated Quality Evaluation of Digital Fundus Photography’ 2009 Acta Ophthalmologica (impact factor: 3.761, # cite: 68)
# > 2022.06.24 jieunoh@postech.ac.kr
# ------------------------------------------------
from illumination import illumination
from sharpness import sharpness
import numpy as np
import os
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# 0. 돌릴때 마다 설정 ---------------------------
image_size = 512
patch_size = 64
path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220623_512_n1000/"
date = '0627'


# 0. 설정 ---------------------------
excel_path ="/home/guest1/ellen_code/RetinaImage_MW/image_evaluation_AQE/AQE_result/input/"# no need to change this
excel_name = "AQE_"+path.split("/")[-2]+"_p" +str(patch_size)+"_"+date+".csv"
plot_name ="AQE_"+path.split("/")[-2]+"_p" +str(patch_size)+"_"+date+".png"

foldersA = ['trainA', 'valA'] # testA는 따로 (l & h 섞여있어서)
foldersB = ['trainB','valB', 'testB']

num = image_size/patch_size
print("-------------------------------")
print("image size: ",image_size)
print("patch_size: ",patch_size)
print("orignal_data: ",path.split("/")[-2])
print("excel_name: ",excel_name)
print("plot_name: ",plot_name)
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게

# df = pd.read_csv(excel_path+excel_name)

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
        quality_scorei=sharpnessi*illuminationi
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
    quality_scorei=sharpnessi*illuminationi
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
        quality_scorei=sharpnessi*illuminationi
        categoryi=category(quality_scorei)

        row =[folder,qualityi, image, sharpnessi, illuminationi, quality_scorei, categoryi]
        df.loc[count] = row
        count+=1
    print(">>>>END: "+folder)
    
# 5-0. Sharpness, Iluumination, Quality score mean& std --------------
roundd = 2
lsm =str(round(df[df["quality"]=='l']["sharpness"].mean(),roundd))
lss =str(round(df[df["quality"]=='l']["sharpness"].std(),roundd))
hsm =str(round(df[df["quality"]=='h']["sharpness"].mean(),roundd))
hss=str(round(df[df["quality"]=='h']["sharpness"].std(),roundd))

lim =str(round(df[df["quality"]=='l']["illumination"].mean(),roundd))
lis =str(round(df[df["quality"]=='l']["illumination"].std(),roundd))
him =str(round(df[df["quality"]=='h']["illumination"].mean(),roundd))
his =str(round(df[df["quality"]=='h']["illumination"].std(),roundd))

lqm =str(round(df[df["quality"]=='l']["quality_score"].mean(),roundd))
lqs =str(round(df[df["quality"]=='l']["quality_score"].std(),roundd))
hqm =str(round(df[df["quality"]=='h']["quality_score"].mean(),roundd))
hqs =str(round(df[df["quality"]=='h']["quality_score"].std(),roundd))

information = "Low-High: s, i ,q \n ("+lsm+", "+lss+") ("+hsm+", "+hss+") \n ("+lim+", "+lis+") ("+him+", "+his+") \n ("+lqm+", "+lqs+") ("+hqm+", "+hqs+")"
# 5. plot -----------------------------
# sharpness

plt.subplot(3,1,1) 
sns.distplot(df[df["quality"] == 'l']["sharpness"], color='red', label = 'low')
sns.distplot(df[df["quality"] == 'h']["sharpness"], color='skyblue', label = 'high ')
plt.legend(title="quality")
# plt.gca().set_title("sharpness")


# illumination
plt.subplot(3,1,2)
sns.distplot(df[df["quality"] == 'l']["illumination"], color='red', label = 'low quality')
sns.distplot(df[df["quality"] == 'h']["illumination"], color='skyblue', label = 'high quality')
# plt.gca().set_title("illumination")



# quality score 
plt.subplot(3,1,3)
sns.distplot(df[df["quality"] == 'l']["quality_score"], color='red', label = 'low quality')
sns.distplot(df[df["quality"] == 'h']["quality_score"], color='skyblue', label = 'high quality')
# plt.gca().set_title("quality_score")

plt.suptitle(information)
plt.tight_layout()
plt.show()
plt.savefig(excel_path+plot_name)

# 6. save it to csv and plot  ---------------------------
df.to_csv(excel_path +excel_name)
print("-----------------------------")
print("cvs saved!!! : "+excel_name)
print("path: "+excel_path+excel_name)
print("plot saved!!! : "+plot_name)
print("path: "+excel_path+plot_name)
print("-----------------------------")
