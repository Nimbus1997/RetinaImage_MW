import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import shutil
#___________
# READ ME-----------------------------------------------------
# 1) train-A,B, test-A,B, val-A,B set으로 나누고
# 2) crop은 X, resize 도 X 그냥 나누기만
# 3) image로 열지 않고 shutil로 파일 복사 붙여넣기
#
# made: 2023.01.09
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
rhq_path = "/root/jieunoh/ellen_data/EyeQ/all_hq"
rlq_path = "/root/jieunoh/ellen_data/EyeQ/all_lq"  # 원본크기
gen_path = "/root/jieunoh/ellen_data/input_eyeq_isecret_total_spilt"
imageformat="jpeg" #원본 이미지 포멧
pre_crop_done = True

trainlR, vallR, testlR = 6, 2, 2  # low quality data의 train : val: test 비율 쓰기
trainhR, valhR, testhR = 6, 2, 2 # high qualtiy data의 train : val: test 비율 쓰기
biggerthanthis = 10000  # 10000보단 작을 것으로 예상
sw = biggerthanthis  # smallst widht 젤작은 width 알아보기 위해서
sh = biggerthanthis  # smallest height 젤 작은 hieght 알아보기 위해서

# 00. seting 된 parameter 확인  ------------------------ 

print("path check")
print("rhq_path: ", rhq_path)
print("rlq_path: ", rlq_path)
print("gen_path: ", gen_path)

print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게




# 0-1. path없으면 path생성 ------------------
folders =["train", "val", "test"]
folders2 = ['crop_good', 'crop_usable','degrade_good']
for f in folders:
    for ff in folders2:
        if not os.path.isdir(gen_path+"/eyeq/"+f+"/"+ff):
            os.makedirs(gen_path+"/eyeq/"+f+"/"+ff)
            print("path generated: ", gen_path+"/eyeq/"+f+"/"+ff)

# 0-2. dataset 나누기 -train val test 개수------------------

# low qualtiy -> A group
ltotal_dataset = len(os.listdir(rlq_path))
ltrainN = int(float(ltotal_dataset)/10.*float(trainlR))
lvalN = int(float(ltotal_dataset)/10.*float(valhR))
ltestN = ltotal_dataset - ltrainN-lvalN



# high qualtiy -> B group & A group in test
htotal_dataset = len(os.listdir(rhq_path))
htrainN = int(float(htotal_dataset)/10.*float(trainhR))
hvalN = int(float(htotal_dataset)/10.*float(valhR))
htestN = htotal_dataset - htrainN-hvalN

print("\n[HQ] ratio of high quality dataset (train : val : test) : (",
      trainhR, ": ", valhR, ": ", testhR, ")")
print("High qualtiy # of dataset in (train : val : test) : (",
      htrainN, ", ", hvalN, ", ", htestN, ")")

print("\n[LQ] ratio of low quality dataset (train : val : test) : (",
      trainlR, ": ", vallR, ": ", testlR, ")")
print("Low qualtiy # of dataset in (train : val : test) : (",
      ltrainN, ", ", lvalN, ", ", ltestN, ")")

print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게


i = 0
step = 0
j=1

# 1. real low quality image  불러오기 [A group] ------------------
folder2= folders2[1]  #crop_usable
for rlq in tqdm(sorted(os.listdir(rlq_path))):
    if imageformat in rlq:
        i+=1
    #2. real lq image 저장하기 ------------------
        if step == 0:
            shutil.copyfile(rlq_path+"/"+rlq, gen_path+"/eyeq/train/"+folder2+"/"+rlq)
        #val set
        if step == 1:
            shutil.copyfile(rlq_path+"/"+rlq, gen_path+"/eyeq/val/"+folder2+"/"+rlq)
        #test set
        if step == 2:
            shutil.copyfile(rlq_path+"/"+rlq, gen_path+"/eyeq/test/"+folder2+"/"+rlq)

        j += 1

        if (i == ltrainN):
            print("\n===["+str(j-1)+" train set end]============================")
            j = 1  # 초기화
            step += 1
            
        elif (i == (ltrainN+lvalN)):
            print("\n===["+str(j-1)+" val set end]============================")
            j = 1
            step += 1
        elif (i==(ltotal_dataset)):
            
            break
        
print("\n===["+str(j-1)+" test set end]============================")
print("[low quality end]===============================================================================\n\n")




# 3. real hq import image [B group] ------------------
i = 0
step = 0
j = 1
folder2= folders2[0]  #crop_good
for rhq in tqdm(sorted(os.listdir(rhq_path))):
    if imageformat in rhq:
        i += 1
    #4. real hq image 저장하기 ------------------
        if step == 0:
            shutil.copyfile(rhq_path+"/"+rhq, gen_path+"/eyeq/train/"+folder2+"/"+rhq)
        #val set
        if step == 1:
            shutil.copyfile(rhq_path+"/"+rhq, gen_path+"/eyeq/val/"+folder2+"/"+rhq)
        #test A set
        if step == 2:
            shutil.copyfile(rhq_path+"/"+rhq, gen_path+"/eyeq/test/"+folder2+"/"+rhq)
        j += 1

        if (i == htrainN):
            print("\n===["+str(j-1)+"train set end]============================")
            j = 1  # 초기화
            step += 1
        elif (i == (htrainN+hvalN)):
            print("\n===["+str(j-1)+"val set end]============================")
            j = 1
            step += 1
        elif (i == htotal_dataset):
            print("\n===["+str(j-1)+"test set end]============================")
            break
    
print("[high quality end]===============================================================================\n\n")




print("smallest (width, height) =  (", sw, ", ", sh, ")")
