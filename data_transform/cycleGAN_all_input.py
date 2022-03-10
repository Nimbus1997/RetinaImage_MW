import os
from re import L
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# ----------------------------------------------------------------
# 1) train-A,B, test-A,B, val-A,B set으로 나누고
# 2) 각각 file 명 1부터 시작하게
# 3) RLQ, RHQ 를 섞어서 B SET만들고 A는 RHQ로만 구성
# --> B: RLQ 다 + RHQ (RLQ개수 만큼) / A: RHQ (남은 RHQ만큼)
# --> 그렇게 구성하면 
#       B: RLQ(910) + RHQ (910) = 1820
#       A: RHQ(2288 - 910) = 1370
#       B>A이긴 하지만 많이 차이는 안나고, A도 1000개 넘으니까 괜찮지 않을까 
# 4) 우선 DATASET을 구성할때 순서대로 읽어와서 TRAIN, VAL, TEST에 넣고 B그룹의 경우 LOW-HIGH하나씩 순서대로 구성할 예정 --> randomize필요하면, 그때
# 
# 참고: AIMI lab meeting/ 20220118_jieunOh_Retina Image.pptx
# 2022.01.11 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
rhq_path = "/home/guest1/ellen_data/UKB_quality_data/high_Q" # real high Q
rlq_path = "/home/guest1/ellen_data/UKB_quality_data/low_Q" #real low Q
gen_path = "/home/guest1/ellen_data/UKB_quality_data/cycleGANinput_all_202220111"
trainR, valR, testR= 6,2,2 # train : val: test 비율 쓰기


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

# 1-1. B group LOW & HIGH quality dataset 개수 나누기
Ltotal_dataset = len(os.listdir(rlq_path))
LtrainN = int(float(Ltotal_dataset)/10.*float(trainR))
LvalN = int(float(Ltotal_dataset)/10.*float(valR)) 
LtestN = Ltotal_dataset - LtrainN-LvalN

print("ratio of dataset (train : val : test) : (", trainR, ": ", valR , ": ", testR ,")")
print("============================================================")
print("[B GROUP] # of LOW QUALITY dataset in (train : val : test) : (", LtrainN, ", ", LvalN , ", ", LtestN ,")")
print("[B GROUP]# of HIGH QUALITY dataset in (train : val : test) : (", LtrainN, ", ", LvalN , ", ", LtestN ,")") # HIGH QUALITY개수와 LOW 동알하므로 각 그룹 개수 동일
print("[B GROUP]# of dataset in (train : val : test) : (", LtrainN*2, ", ", LvalN*2 , ", ", LtestN*2 ,")") 
print("============================================================")

# 1-2. A group HIGH quality dataset 개수 나누기
Htotal_dataset = len(os.listdir(rhq_path))
HAtotal_dataset = Htotal_dataset-Ltotal_dataset # High quality A group dataset
HAtrainN = int(float(HAtotal_dataset)/10.*float(trainR))
HAvalN = int(float(HAtotal_dataset)/10.*float(valR)) 
HAtestN = HAtotal_dataset - HAtrainN-HAvalN
print("[A GROUP]# of HIGH QUALITY dataset in (train : val : test) : (", HAtrainN, ", ", HAvalN , ", ", HAtestN ,")")
print("============================================================")
print("============================================================")

i=1 # 개수 
step=0
j=1 # image번호
# 2. B group 구성 ------------------------------------
# 2-1. low qaulity - 홀수번호------------------
for rlq in  os.listdir(rlq_path):
    # 2-1-1. 이미지 읽기
    rlqimg = Image.open(rlq_path+"/"+rlq)
    # 2-1-2. 이미지 저장
    if step ==0 : 
        rlqimg.save(gen_path+"/trainB/"+str(j)+".jpg")
        print("["+str(i)+"] trainB saved")
    #val set
    if step ==1 : 
        rlqimg.save(gen_path+"/valB/"+str(j)+".jpg")
        print("["+str(i)+"] valB saved")
    #test set
    if step ==2 : 
        rlqimg.save(gen_path+"/testB/"+str(j)+".jpg")
        print("["+str(i)+"] testB saved")

    j+=2 #low quality 이미지는 홀수만 구성 ==> 2씩 증가

    # 2-1-3. train, val, test step update
    if (i==LtrainN):
        j=1 #초기화
        step+=1
        print("===[train set end]============================")
    if (i==(LtrainN+LvalN)):
        j=1
        step+=1
        print("===[val set end]============================")
    
    i+=1 # 이미지읽은 개수 

# 2-2. high qaulity ------------------
i=1 # 개수 
group = 0 
step=0
j=2 # image번호
for rhq in  os.listdir(rhq_path):

    # 2-2. high qaulity: 짝수 번호 ------------------
    if group ==0 : 
        # 2-1-1. 이미지 읽기
        rhqimg = Image.open(rhq_path+"/"+rhq)
        # 2-1-2. 이미지 저장
        if step ==0 : 
            rhqimg.save(gen_path+"/trainB/"+str(j)+".jpg")
            print("["+str(j)+"] trainB saved")
        #val set
        if step ==1 : 
            rhqimg.save(gen_path+"/valB/"+str(j)+".jpg")
            print("["+str(j)+"] valB saved")
        #test set
        if step ==2 : 
            rhqimg.save(gen_path+"/testB/"+str(j)+".jpg")
            print("["+str(j)+"] testB saved")

        j+=2 #high quality 이미지는 짝수만 구성 ==> 2씩 증가
        
        # 2-1-3. train, val, test step update
        if (i==LtrainN):
            j=2 #초기화
            step+=1
            print("===[train set end]============================")
        if (i==(LtrainN+LvalN)):
            j=2
            step+=1
            print("===[val set end]============================")
    
        
        if i == Ltotal_dataset: # B GROUP 끝
            group =1
            i =1 # 개수 초기화
            j=1 # 파일명 초기화
            step =0 # 초기화
        
        
    # 3. A group - high quality data -------------------------------------------
    if group ==1: 
        # 3-1. 이미지 읽기  -------------------
        rhqimg = Image.open(rhq_path+"/"+rhq)
        # 3-2. 이미지 저장  -------------------
        if step ==0 : 
            rhqimg.save(gen_path+"/trainA/"+str(j)+".jpg")
            print("["+str(j)+"] trainA saved")
        #val set
        if step ==1 : 
            rhqimg.save(gen_path+"/valA/"+str(j)+".jpg")
            print("["+str(j)+"] valA saved")
        #test set
        if step ==2 : 
            rhqimg.save(gen_path+"/testA/"+str(j)+".jpg")
            print("["+str(j)+"] testA saved")

        j+=1 #다시 1씩 증가
        
        # 3-3. train, val, test step update  -------------------
        if (i==HAtrainN):
            j=1 #초기화
            step+=1
            print("===[train set end]============================")
        if (i==(HAtrainN+HAvalN)):
            j=1
            step+=1
            print("===[val set end]============================")
    
        i+=1 # 이미지 읽은 개수 

