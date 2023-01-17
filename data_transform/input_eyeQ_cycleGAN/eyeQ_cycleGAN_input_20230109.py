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
gen_path = "/root/jieunoh/ellen_data/input_eyeq_total_spilt"
imageformat="jpeg" #원본 이미지 포멧


# ------------------------

# low quality: group A
# high quality: group B
trainlR, vallR, testlR = 6, 2, 2  # low quality data의 train : val: test 비율 쓰기
trainhR, valhR, testhaR, testhbR = 6, 2, 1, 1 # high qualtiy data의 train : val: test 비율 쓰기
biggerthanthis = 10000  # 10000보단 작을 것으로 예상
sw = biggerthanthis  # smallst widht 젤작은 width 알아보기 위해서
sh = biggerthanthis  # smallest height 젤 작은 hieght 알아보기 위해서

# 00. seting 된 parameter 확인  ------------------------ 

print("path check")
print("rlq_path: ", rlq_path)
print("rhq_path: ", rhq_path)
print("gen_path: ", gen_path)
print("원본 이미지 포멧: ", imageformat)
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게



# 0-1. path없으면 path생성 ------------------
if not os.path.isdir(gen_path+"/trainA"):
    os.makedirs(gen_path+"/trainA")
    print("train path generated!")
if not os.path.isdir(gen_path+"/valA"):
    os.makedirs(gen_path+"/valA")
    print("val path generated!")
if not os.path.isdir(gen_path+"/testA"):
    os.makedirs(gen_path+"/testA")
    print("test path generated!")

if not os.path.isdir(gen_path+"/trainB"):
    os.makedirs(gen_path+"/trainB")
    print("train path generated!")
if not os.path.isdir(gen_path+"/valB"):
    os.makedirs(gen_path+"/valB")
    print("val path generated!")
if not os.path.isdir(gen_path+"/testB"):
    os.makedirs(gen_path+"/testB")
    print("test path generated!")


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
htestaN = int(float(htotal_dataset)/10.*float(testhaR))
htestbN = htotal_dataset - htrainN-hvalN-htestaN





print("\n[A group] ratio of low quality dataset (train : val : test) : (",
      trainlR, ": ", vallR, ": ", testlR, ")")
print("Low qualtiy # of dataset in (train : val : test) : (",
      ltrainN, ", ", lvalN, ", ", ltestN, ")")


print("\n[B group] ratio of high quality dataset (train : val : test-A: test-B) : (",
      trainhR, ": ", valhR, ": ", testhaR, ": ", testhbR, ")")
print("High qualtiy # of dataset in (train : val : test-A: test-B) : (",
      htrainN, ", ", hvalN, ", ", htestaN, ", ", htestbN, ")")
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게


i = 0
step = 0
j = 1

# 1. real low quality image  불러오기 [A group] ------------------
for rlq in tqdm(sorted(os.listdir(rlq_path))):
    if imageformat in rlq:
        i += 1
    #2. real lq image 저장하기 ------------------
        if step == 0:
            shutil.copyfile(rlq_path+"/"+rlq, gen_path+"/trainA/"+rlq)
            # print("["+str(i)+"] trainA saved")
        #val set
        if step == 1:
            shutil.copyfile(rlq_path+"/"+rlq, gen_path+"/valA/"+rlq)
            # print("["+str(i)+"] valA saved")
        #test set
        if step == 2:
            shutil.copyfile(rlq_path+"/"+rlq, gen_path+"/testA/"+rlq)
            # print("["+str(i)+"] testA saved")

        j += 1

        if (i == ltrainN):
            print("===["+str(j-1)+" train set end]============================")
            j = 1  # 초기화
            step += 1
            
        elif (i == (ltrainN+lvalN)):
            print("===["+str(j-1)+" val set end]============================")
            j = 1
            step += 1
        elif (i==(ltotal_dataset)):
            step=4
            break
        
print("===["+str(j-1)+" test set end]============================")
print("[low quality end]===============================================================================\n\n")




# 3. real hq import image [B group] ------------------
i = 0
step = 0
j = 1
for rhq in tqdm(sorted(os.listdir(rhq_path))):
    if imageformat in rhq:
        i += 1
    #4. real hq image 저장하기 ------------------
        if step == 0:
            shutil.copyfile(rhq_path+"/"+rhq, gen_path+"/trainB/"+rhq)
            # print("["+str(i)+"] trainB saved")
        #val set
        if step == 1:
            shutil.copyfile(rhq_path+"/"+rhq, gen_path+"/valB/"+rhq)
        #test A set
        if step == 2:
            shutil.copyfile(rhq_path+"/"+rhq, gen_path+"/testA/"+rhq)
        #test B set
        if step == 3:
            shutil.copyfile(rhq_path+"/"+rhq, gen_path+"/testB/"+rhq)
        j += 1

        if (i == htrainN):
            print("===["+str(j-1)+"train set end]============================")
            j = 1  # 초기화
            step += 1
        elif (i == (htrainN+hvalN)):
            print("===["+str(j-1)+"val set end]============================")
            j = 1
            step += 1
        elif (i == (htrainN+hvalN+htestaN)):
            print("===["+str(j-1)+"testA set end]============================")
            j = 1
            step += 1
        elif i ==htotal_dataset:
            step=4
            break

print("===["+str(j-1)+"testB set end]============================")
print("[high quality end]===============================================================================\n\n")

