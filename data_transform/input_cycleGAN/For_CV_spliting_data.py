"""
 [input_20220623_#_n1000]를 for Cross Validation을 위해서 
 PART 2
 ---PART 1 []----------------------
 1. hq와 lq모우기-> *_hq , *_lq 폴더에 생성 0001부터 - 1000까지

 ---PART 1 [ V ]----------------------
  2. 3 cross val 만들기 -> based "cycleGAN_rlqrhq_input_20220622.py"
    (1) 원래처럼 train- val -test 순 [이미 완료]
    (2) test-train-val 순
    (3) val-test-train 순
----------------------------------
 
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


# 0. augment 설정 ------------------
rhq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220623_512_n1000_hq"
rlq_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220623_512_n1000_lq" 
path2="input_20221211_2_512_n1000"
path3="input_20221211_3_512_n1000"
gen_path2 = "/home/guest1/ellen_data/UKB_quality_data2_combined/"+path2
gen_path3 = "/home/guest1/ellen_data/UKB_quality_data2_combined/"+path3

imageformat="jpg" #원본 이미지 포멧

# total image개수 정하고 싶으면------------
set_total = True
total_img_sample = 1000 #l, h each 
# ------------------------



trainlR, vallR, testlR = 6, 2, 2  # low quality data의 train : val: test 비율 쓰기
trainhR, valhR, testhaR, testhbR = 6, 2, 1, 1 # high qualtiy data의 train : val: test 비율 쓰기
biggerthanthis = 10000  # 10000보단 작을 것으로 예상
sw = biggerthanthis  # smallst widht 젤작은 width 알아보기 위해서
sh = biggerthanthis  # smallest height 젤 작은 hieght 알아보기 위해서

# 00. seting 된 parameter 확인  ----------------------------------------------------------------------------------------- 

print("path check")
print("rlq_path: ", rlq_path)
print("rhq_path: ", rhq_path)
print("gen_path2: ", gen_path2)
print("gen_path3: ", gen_path3)

print("원본 이미지 포멧: ", imageformat)
print("total_img_sample: ", set_total,",", total_img_sample)

print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게



# 0-1. path없으면 path생성 ------------------------------------------------------------------------------------------
for gen_path in [gen_path2, gen_path3]:
    print(gen_path)
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

# 0-2. dataset 나누기 -train val test 개수------------------------------------------------------------------------------------------------------------

# low qualtiy -> A group
if set_total:
    ltotal_dataset=total_img_sample
else:
    ltotal_dataset = len(os.listdir(rlq_path))
ltrainN = int(float(ltotal_dataset)/10.*float(trainlR))
lvalN = int(float(ltotal_dataset)/10.*float(valhR))
ltestN = ltotal_dataset - ltrainN-lvalN

ldigit = len(str(ltotal_dataset)) # 자릿수
print("\n-------------------------------")
print("ldigit: ", ldigit)

# high qualtiy -> B group & A group in test
if set_total:
    htotal_dataset=total_img_sample
else:
    htotal_dataset = len(os.listdir(rhq_path))
htrainN = int(float(htotal_dataset)/10.*float(trainhR))
hvalN = int(float(htotal_dataset)/10.*float(valhR))
htestaN = int(float(htotal_dataset)/10.*float(testhaR)) #Agroup
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



# [PATH2] ------------------------------------------------------------------------------------------------------------------------------
# 1. real low quality image  불러오기 [A group] -------------------------------
print("------------------------------------------------------")
print("--[",path2,"]-----------------------------------------")
i = 0
step = 0
j = 1
for rlq in tqdm(sorted(os.listdir(rlq_path))):
    if imageformat in rlq:
        i += 1
        rlqimg = Image.open(rlq_path+"/"+rlq)

    #2. real lq image 저장하기 ------------------
        if step == 0:
            rlqimg.save(gen_path2+"/testA/"+rlq)
            j += 1
        #val set
        if step == 1:
            rlqimg.save(gen_path2+"/trainA/"+rlq)
            j += 1
        #test set
        if step == 2:
            rlqimg.save(gen_path2+"/valA/"+rlq)
            j += 1

        

        if (i == ltestN):
            print("===["+str(j-1)+" testA set end]============================")
            j = 1  # 초기화
            step += 1
            
        elif (i == (ltrainN+ltestN)):
            print("===["+str(j-1)+" trainA set end]============================")
            j = 1
            step += 1
        elif (i==(ltotal_dataset)):
            step=4
            break
        
print("===["+str(j-1)+" valA set end]============================")
print("[low quality end]==================================\n\n")



# 3. real hq import image [B group] ------------------
i = 0
step = 0
j = 1
for rhq in tqdm(sorted(os.listdir(rhq_path))):
    if imageformat in rhq:
        rhqimg = Image.open(rhq_path+"/"+rhq)
        i += 1


    #4. real hq image 저장하기 ------------------
        if step == 0:
            rhqimg.save(gen_path2+"/testA/h"+rhq)
            j += 1
        #val set
        if step == 1:
            rhqimg.save(gen_path2+"/testB/"+rhq)
            j += 1
        #test A set
        if step == 2:
            rhqimg.save(gen_path2+"/trainB/"+rhq) 
            j += 1
        #test B set
        if step == 3:
            rhqimg.save(gen_path2+"/valB/"+rhq)
            j += 1


        if (i == htestaN):
            print("===["+str(j-1)+"testA set end]============================")
            j = 1  # 초기화
            step += 1
        elif (i == (htestaN+htestbN)):
            print("===["+str(j-1)+"testB set end]============================")
            j = 1
            step += 1
        elif (i == (htrainN+htestaN+htestbN)):
            print("===["+str(j-1)+"trainB set end]============================")
            j = 1
            step += 1
        elif i ==htotal_dataset:
            step=4
            break

print("===["+str(j-1)+"valB set end]============================")
print("[high quality end]======================================\n\n")

# # [PATH3] -val-test-train 순-----------------------------------------------------------------------------------------------------------------------------

# print("------------------------------------------------------")
# print("--[",path3,"]-----------------------------------------")
# i = 0
# step = 0
# j = 1
# for rlq in tqdm(sorted(os.listdir(rlq_path))):
#     if imageformat in rlq:
#         i += 1
#         rlqimg = Image.open(rlq_path+"/"+rlq)

#     #2. real lq image 저장하기 ------------------
#         if step == 0:
#             rlqimg.save(gen_path3+"/valA/"+rlq)
#             j += 1
#         #val set
#         if step == 1:
#             rlqimg.save(gen_path3+"/testA/"+rlq)
#             j += 1
#         #test set
#         if step == 2:
#             rlqimg.save(gen_path3+"/trainA/"+rlq)
#             j += 1

#         if (i == lvalN):
#             print("===["+str(j-1)+" valA set end]============================")
#             j = 1  # 초기화
#             step += 1
            
#         elif (i == (lvalN+ltestN)):
#             print("===["+str(j-1)+" testA set end]============================")
#             j = 1
#             step += 1
#         elif (i==(ltotal_dataset)):
#             step=4
#             break
        
# print("===["+str(j-1)+" trainA set end]============================")
# print("[low quality end]==================================\n\n")





# # 3. real hq import image [B group] ------------------
# i = 0
# step = 0
# j = 1
# for rhq in tqdm(sorted(os.listdir(rhq_path))):
#     if imageformat in rhq:
#         rhqimg = Image.open(rhq_path+"/"+rhq)
#         i += 1


#     #4. real hq image 저장하기 ------------------
#         if step == 0:
#             rhqimg.save(gen_path3+"/valB/"+rhq)
#             j += 1
#         #val set
#         if step == 1:
#             rhqimg.save(gen_path3+"/testA/h"+rhq)
#             j += 1
#         #test A set
#         if step == 2:
#             rhqimg.save(gen_path3+"/testB/"+rhq) 
#             j += 1
#         #test B set
#         if step == 3:
#             rhqimg.save(gen_path3+"/trainB/"+rhq)
#             j += 1

        

#         if (i == hvalN):
#             print("===["+str(j-1)+"valB set end]============================")
#             j = 1  # 초기화
#             step += 1
#         elif (i == (htestaN+hvalN)):
#             print("===["+str(j-1)+"testA set end]============================")
#             j = 1
#             step += 1
#         elif (i == (hvalN+htestaN+htestbN)):
#             print("===["+str(j-1)+"testB set end]============================")
#             j = 1
#             step += 1
#         elif i ==htotal_dataset:
#             step=4
#             break

# print("===["+str(j-1)+"trainB set end]============================")
# print("[high quality end]======================================\n\n")