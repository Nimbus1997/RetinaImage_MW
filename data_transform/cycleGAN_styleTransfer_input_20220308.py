import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

#___________
# READ ME-----------------------------------------------------
# 1) train-A,B, test-A,B, val-A,B set으로 나누고
# 2) 각각 file 명은 "id(폴더명)_현재이름" 으로 구성
# 3) Canon이 제일 좋은듯 (혈관 잘보임) ==> B로 구성 (A to B)
# 4) 데이터 구성 - 우선 다 A: topcon, opto // B: canon으로 구성
#    train&val (A) : topcon, opto
#    train&bal (B) : canon
#    test (A)  : topcon, opto
#    test (B)  : canon
#
# 2022.03.08 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
rhq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/high_Q"
rlq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/low_Q"  # 원본크기
gen_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_202202204545"
imageformat="png" #원본 이미지 포멧
biggerthanthis = 10000  # 10000보단 작을 것으로 예상
sw = biggerthanthis  # smallst widht 젤작은 width 알아보기 위해서
sh = biggerthanthis  # smallest height 젤 작은 hieght 알아보기 위해서
trainlR, vallR, testlR = 6, 2, 2  # low quality data의 train : val: test 비율 쓰기
# high qualtiy data의 train : val: test 비율 쓰기
trainhR, valhR, testhaR, testhbR = 6, 2, 0.5, 1.5

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

ldigit = len(str(ltotal_dataset)) # 자릿수
print("\n-------------------------------")
print("ldigit: ", ldigit)

# high qualtiy -> B group & A group in test
htotal_dataset = len(os.listdir(rhq_path))
htrainN = int(float(htotal_dataset)/10.*float(trainhR))
hvalN = int(float(htotal_dataset)/10.*float(valhR))
htestaN = int(float(htotal_dataset)/10.*float(testhaR))
htestbN = htotal_dataset - htrainN-hvalN-htestaN

hdigit = len(str(htotal_dataset)) # 자릿수
print("hdigit: ", hdigit)



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
for rlq in os.listdir(rlq_path):
    if imageformat in rlq:
        rlqimg = Image.open(rlq_path+"/"+rlq)
        i += 1
        w, h = rlqimg.size
        if sw > w:
            sw = w
        if sh > h:
            sh = h

    #2. real lq image 저장하기 ------------------
        if step == 0:
            rlqimg.save(gen_path+"/trainA/"+str(j).zfill(ldigit)+".jpg")
            # print("["+str(i)+"] trainA saved")
        #val set
        if step == 1:
            rlqimg.save(gen_path+"/valA/"+str(j).zfill(ldigit)+".jpg")
            # print("["+str(i)+"] valA saved")
        #test set
        if step == 2:
            rlqimg.save(gen_path+"/testA/"+str(j).zfill(ldigit)+".jpg")
            # print("["+str(i)+"] testA saved")

        j += 1

        if (i == ltrainN):
            print("===["+str(j-1)+" train set end]============================")
            j = 1  # 초기화
            step += 1
            
        if (i == (ltrainN+lvalN)):
            print("===["+str(j-1)+" val set end]============================")
            j = 1
            step += 1
        
print("===["+str(j-1)+" test set end]============================")
print("[low quality end]===============================================================================\n\n")




# 3. real hq import image [B group] ------------------
i = 0
step = 0
j = 1
for rhq in os.listdir(rhq_path):
    if imageformat in rhq:
        rhqimg = Image.open(rhq_path+"/"+rhq)
        i += 1
        if sw > w:
            sw = w
        if sh > h:
            sh = h

    #4. real hq image 저장하기 ------------------
        if step == 0:
            rhqimg.save(gen_path+"/trainB/"+str(j).zfill(hdigit)+".jpg")
            # print("["+str(i)+"] trainB saved")
        #val set
        if step == 1:
            rhqimg.save(gen_path+"/valB/"+str(j).zfill(hdigit)+".jpg")
            # print("["+str(i)+"] valB saved")
        #test A set
        if step == 2:
            rhqimg.save(gen_path+"/testA/h"+str(j).zfill(hdigit)+".jpg") # high quality image는 h붙여서 저장 A group test에서 
            # print("["+str(i)+"] testB saved")
        #test B set
        if step == 3:
            rhqimg.save(gen_path+"/testB/"+str(j).zfill(hdigit)+".jpg")
            # print("["+str(i)+"] testB saved")

        j += 1

        if (i == htrainN):
            print("===["+str(j-1)+"train set end]============================")
            j = 1  # 초기화
            step += 1
        if (i == (htrainN+hvalN)):
            print("===["+str(j-1)+"val set end]============================")
            j = 1
            step += 1
        if (i == (htrainN+hvalN+htestaN)):
            print("===["+str(j-1)+"testA set end]============================")
            j = 1
            step += 1

print("===["+str(j)+"testB set end]============================")
print("[high quality end]===============================================================================\n\n")




print("smallest (width, height) =  (", sw, ", ", sh, ")")
