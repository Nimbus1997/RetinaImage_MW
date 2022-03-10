import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

#___________
# READ ME-----------------------------------------------------
# 0) 이 코드를 바탕으로 새로운 코드 만들때 "check_ellen"검색해서 주의깊게 보기
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
ori_path = "/home/guest1/ellen_data/SERI2_2_variousCamera"
gen_path = "/home/guest1/ellen_data/SERI2_2_variousCamera_20220310"
agroupCamera="canon" #canon, op_(또는 opto), topcon 중에서 
imageformat = ["jpg", "JPG"]  # 원본 이미지 포멧
smallest_willbe_bigger_than_this = 10000  # 이미지 사이즈 10000보단 작을 것으로 예상
sw = smallest_willbe_bigger_than_this  # smallst widht 젤작은 width 알아보기 위해서
sh = smallest_willbe_bigger_than_this  # smallest height 젤 작은 hieght 알아보기 위해서
btrainR, bvalR, btestR = 6, 2, 2  # low quality data의 train : val: test 비율 쓰기
atrainR, avalR, atestR, atestbR = 6, 2, 1, 1 # high qualtiy data의 train : val: test : test(b) 비율 쓰기 
# A와 B group의 train & val의 R는 같게. 결국 patient 수는 같게 되게하기 
# atestbR는 정답이 되는 agroup의 이미지를 input으로 받았을 때는 그대로 잘 a를 output하는지 알아보기 위해서

print("path check")
print("ori_path: ", ori_path)
print("rhq_path: ", gen_path)
print("원본 이미지 포멧: ", imageformat)
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>")  # 확인후 넘어가게 입력되면 넘어가게


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
N_patient = len(os.listdir(ori_path))-1 # .DS_Store이라는 폴더가 들어가있어서 -1함 check_ellen
#B group에 들어갈 폴더수
btrainN = int(float(N_patient)/10.*float(btrainR)) *2
bvalN = int(float(N_patient)/10.*float(bvalR)) *2
btestN = N_patient*2 - btrainN-bvalN

# A그룹에 들어갈 폴더수
atrainN = int(float(N_patient)/10.*float(atrainR))
avalN = int(float(N_patient)/10.*float(avalR))
atestN = int(float(N_patient)/10.*float(atestR))
atestbN = N_patient - atrainN-avalN-atestN

print("\n[A group] ratio of <", agroupCamera ,"> (train : val : test : testB) : (",
      atrainR, ", ", avalR, ", ", atestR, ": ",atestbR,")")
print("[A group] # of dataset <", agroupCamera ,"> (train : val : test : testB) : (",
      atrainN, ": ", avalN, ": ", atestN,": ",atestbN, ")")

print("\n[B group] ratio of other camera  (train : val : test) : (",
      btrainR, ": ", bvalR, ": ", btestR,  ")")
print("[B group] # of dataset of other cameras (train : val : test) : (",
      btrainN, ", ", bvalN, ", ", btestN, ")")
print("B group은 카메라가 두대이기 때문에 개수가 2배임")

print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>")  # 확인후 넘어가게 입력되면 넘어가게

i = 0 # 이미지 개수 세기
step = 0
j = 1 # 각 step에서 image 개수 세기

# 1. [A group] image  불러오기  ------------------
for patient_id in os.listdir(ori_path):
    for image in os.listdir(patient_id): 
        if (imageformat[0] in image or imageformat[1] in image) and agroupCamera in image:
            aimg = Image.open(ori_path+"/"+patient_id+"/"+image)
            i += 1
            w, h = aimg.size
            if sw > w:
                sw = w
            if sh > h:
                sh = h

    #2. real lq image 저장하기 ------------------
        if step == 0:
            aimg.save(gen_path+"/trainA/"+patient_id+"_"+image.split(".")[0]+".jpg")
            # print("["+str(i)+"] trainA saved")
        #val set
        if step == 1:
            aimg.save(gen_path+"/valA/"+patient_id+"_"+image.split(".")[0]+".jpg")
            # print("["+str(i)+"] valA saved")
        #test set
        if step == 2:
            aimg.save(gen_path+"/testA/"+patient_id+"_"+image.split(".")[0]+".jpg")
            # print("["+str(i)+"] testA saved")
        if step ==3: 
            aimg.save(gen_path+"/testB/"+patient_id+"_"+image.split(".")[0]+".jpg")
            
        j += 1

        if (i == atrainN):
            print("===["+str(j-1)+" train set end]============================")
            j = 1  # 초기화
            step += 1

        if (i == (atrainN+avalN)):
            print("===["+str(j-1)+" val set end]============================")
            j = 1
            step += 1
        
        if ( i==(atrainN+avalN+atestN)):
            print("===["+str(j-1)+" test set end]============================")
            j=1
            step +=1

print("===["+str(j)+"testB set end]============================")
print("[A group END]===============================================================================\n\n")


# 3. [B group] image  ------------------
i = 0
step = 0
j = 1
for patient_id in os.listdir(ori_path):
    for image in os.listdir(patient_id): 
        if (imageformat[0] in image or imageformat[1] in image) and agroupCamera not in image:
            aimg = Image.open(ori_path+"/"+patient_id+"/"+image)
            i += 1
            w, h = aimg.size
            if sw > w:
                sw = w
            if sh > h:
                sh = h

    #2. real lq image 저장하기 ------------------
        if step == 0:
            aimg.save(gen_path+"/trainB/"+patient_id+"_"+image.split(".")[0]+".jpg")
            # print("["+str(i)+"] trainA saved")
        #val set
        if step == 1:
            aimg.save(gen_path+"/valB/"+patient_id+"_"+image.split(".")[0]+".jpg")
            # print("["+str(i)+"] valA saved")
        #test set
        if step == 2:
            aimg.save(gen_path+"/testB/"+patient_id+"_"+image.split(".")[0]+".jpg")
            # print("["+str(i)+"] testA saved")
            
        j += 1

        if (i == atrainN):
            print("===["+str(j-1)+" train set end]============================")
            j = 1  # 초기화
            step += 1

        if (i == (atrainN+avalN)):
            print("===["+str(j-1)+" val set end]============================")
            j = 1
            step += 1
    
print("===["+str(j-1)+" test set end]============================")
print("[B group END]===============================================================================\n\n")


print("smallest (width, height) =  (", sw, ", ", sh, ")")
