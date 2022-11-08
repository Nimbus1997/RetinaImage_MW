import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

#___________
# READ ME-----------------------------------------------------
# 1) train, test, val로 나눔
# 2) 각 folder안은 crop_good(hq), crop_usable(lq)로 나누기
# 3) hq, lq 1000개씩 만들기 6:2:2 (train:test:val)
#
# 2022.11.07 jieunoh@postech.ac.kr
# adjust 2022.06.22 -> total_image sample을 정할 수 있음 & size도 한번에 정하기
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
rhq_path = "/home/ellen/ellen_data/EyeQ/all_hq_crop"
rlq_path = "/home/ellen/ellen_data/EyeQ/all_lq_crop"  # 원본크기
gen_path = "/home/ellen/ellen_data/input_eyeq_20221107_512_n1000"
imageformat="png" #원본 이미지 포멧

# total image개수 정하고 싶으면------------
set_total = True
total_img_sample = 1000 #l, h each 
# ------------------------
resize= 512


trainlR, vallR, testlR = 6, 2, 2  # low quality data의 train : val: test 비율 쓰기
trainhR, valhR, testhR = 6, 2, 2 # high qualtiy data의 train : val: test 비율 쓰기
biggerthanthis = 10000  # 10000보단 작을 것으로 예상
sw = biggerthanthis  # smallst widht 젤작은 width 알아보기 위해서
sh = biggerthanthis  # smallest height 젤 작은 hieght 알아보기 위해서

# 00. seting 된 parameter 확인  ------------------------ 

print("path check")
print("rlq_path: ", rlq_path)
print("rhq_path: ", rhq_path)
print("gen_path: ", gen_path)
print("원본 이미지 포멧: ", imageformat)
print("total_img_sample: ", set_total,",", total_img_sample)
print("resize: ", resize)

print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게



# 0-1. path없으면 path생성 ------------------
folders =["train", "val", "test"]
folders2 = ['crop_good', 'crop_usable','degrade_good','gt_cropgood_resize']
for f in folders:
    for ff in folders2:
        if not os.path.isdir(gen_path+"/"+f+"/"+ff):
            os.makedirs(gen_path+"/"+f+"/"+ff)
            print("path generated: ", gen_path+"/"+f+"/"+ff)

# 0-2. dataset 나누기 -train val test 개수------------------

# low qualtiy -> A group
if set_total:
    ltotal_dataset=total_img_sample
else:
    ltotal_dataset = len(os.listdir(rlq_path))
ltrainN = int(float(ltotal_dataset)/10.*float(trainlR))
lvalN = int(float(ltotal_dataset)/10.*float(valhR))
ltestN = ltotal_dataset - ltrainN-lvalN



# high qualtiy -> B group & A group in test
if set_total:
    htotal_dataset=total_img_sample
else:
    htotal_dataset = len(os.listdir(rhq_path))
htrainN = int(float(htotal_dataset)/10.*float(trainhR))
hvalN = int(float(htotal_dataset)/10.*float(valhR))
htestN = htotal_dataset - htrainN-hvalN


print("\n[LQ] ratio of low quality dataset (train : val : test) : (",
      trainlR, ": ", vallR, ": ", testlR, ")")
print("Low qualtiy # of dataset in (train : val : test) : (",
      ltrainN, ", ", lvalN, ", ", ltestN, ")")


print("\n[HQ] ratio of high quality dataset (train : val : test) : (",
      trainhR, ": ", valhR, ": ", testhR, ")")
print("High qualtiy # of dataset in (train : val : test) : (",
      htrainN, ", ", hvalN, ", ", htestN, ")")
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>") # 확인후 넘어가게 입력되면 넘어가게


i = 0
step = 0
j=1

# 1. real low quality image  불러오기 [A group] ------------------
folder2= folders2[0]  #crop_good
for rlq in tqdm(os.listdir(rlq_path)):
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
            img_resize = rlqimg.resize((resize,resize),Image.LANCZOS)
            img_resize.save(gen_path+"/train/"+folder2+"/"+rlq)
            # print("["+str(i)+"] trainA saved")
        #val set
        if step == 1:
            img_resize = rlqimg.resize((resize,resize),Image.LANCZOS)
            img_resize.save(gen_path+"/val/"+folder2+"/"+rlq)
            # print("["+str(i)+"] valA saved")
        #test set
        if step == 2:
            img_resize = rlqimg.resize((resize,resize),Image.LANCZOS)
            img_resize.save(gen_path+"/test/"+folder2+"/"+rlq)
            # print("["+str(i)+"] testA saved")

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
folder2= folders2[1]  #crop_usable
for rhq in tqdm(os.listdir(rhq_path)):
    if imageformat in rhq:
        rhqimg = Image.open(rhq_path+"/"+rhq)
        i += 1
        if sw > w:
            sw = w
        if sh > h:
            sh = h

    #4. real hq image 저장하기 ------------------
        if step == 0:
            img_resize = rhqimg.resize((resize,resize),Image.LANCZOS)
            img_resize.save(gen_path+"/train/"+folder2+"/"+rhq)
            # print("["+str(i)+"] trainB saved")
        #val set
        if step == 1:
            img_resize = rhqimg.resize((resize,resize),Image.LANCZOS)
            img_resize.save(gen_path+"/val/"+folder2+"/"+rhq)
            # print("["+str(i)+"] valB saved")
        #test A set
        if step == 2:
            img_resize = rhqimg.resize((resize,resize),Image.LANCZOS)
            img_resize.save(gen_path+"/test/"+folder2+"/"+rhq) # high quality image는 h붙여서 저장 A group test에서 
            # print("["+str(i)+"] testB saved")
        
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
