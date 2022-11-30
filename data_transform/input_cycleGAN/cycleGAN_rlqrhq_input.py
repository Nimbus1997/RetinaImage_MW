import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# ----------------------------------------------------------------
# 1) train-A,B, test-A,B, val-A,B set으로 나누고
# 2) 각각 file 명 1부터 시작하게
# 3) RLQ로 B SET만들고 A는 RHQ로 구성
# --> pix2pix 과 다른것은 원래 크기대로. --> 원래 크기대로하면 크기 안맞아서 문제 될 수 있는데, 문제되면 그때 고치겠음
# 
# 참고: https://copycoding.tistory.com/159 
# 2022.01.10 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
rhq_path = "/home/guest1/ellen_data/from_paper_20220121/no_artifact"
rlq_path = "/home/guest1/ellen_data/from_paper_20220121/artifacts" #원본크기
gen_path = "/home/guest1/ellen_data/from_paper_20220121/cycleGANinput_rhqrlq_20220121_paper"
biggerthanthis=10000 # 10000보단 작을 것으로 예상
sw=biggerthanthis # 젤작은 width 알아보기 위해서
sh=biggerthanthis # 젤 작은 hieght 알아보기 위해서
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


# 0-2. dataset 나누기 -train val test ------------------
# high qualtiy
htotal_dataset = len(os.listdir(rhq_path))
htrainN = int(float(htotal_dataset)/10.*float(trainR))
hvalN = int(float(htotal_dataset)/10.*float(valR)) 
htestN = htotal_dataset - htrainN-hvalN
# low qualtiy
ltotal_dataset = len(os.listdir(rlq_path))
ltrainN = int(float(ltotal_dataset)/10.*float(trainR))
lvalN = int(float(ltotal_dataset)/10.*float(valR)) 
ltestN = ltotal_dataset -ltrainN-lvalN

print("ratio of dataset (train : val : test) : (", trainR, ": ", valR , ": ", testR ,")")
print("High qualtiy # of dataset in (train : val : test) : (", htrainN, ", ", hvalN , ", ", htestN ,")")
print("Low qualtiy # of dataset in (train : val : test) : (", ltrainN, ", ", lvalN , ", ", ltestN ,")")


i=0
step=0
j=1


# 1. real hq import image ------------------
for rhq in os.listdir(rhq_path):

    rhqimg = Image.open(rhq_path+"/"+rhq)
    w,h=rhqimg.size
    i+=1
    if sw>w:
        sw=w
    if sh>h:
        sh=h
    
   #2. real hq image 저장하기 ------------------
    if step ==0 : 
        rhqimg.save(gen_path+"/trainB/"+str(j)+".jpg")
        print("["+str(i)+"] trainB saved")
    #val set
    if step ==1 : 
        rhqimg.save(gen_path+"/valB/"+str(j)+".jpg")
        print("["+str(i)+"] valB saved")
    #test set
    if step ==2 : 
        rhqimg.save(gen_path+"/testB/"+str(j)+".jpg")
        print("["+str(i)+"] testB saved")

    j+=1
    
    if (i==htrainN):
        j=1 #초기화
        step+=1
        print("===[train set end]============================")
    if (i==(htrainN+hvalN)):
        j=1
        step+=1
        print("===[val set end]============================")

print("[high quality end]===============================================================================\n\n")




# 3. real low quality image  불러오기 ------------------
i=0
step=0
j=1

for rlq in os.listdir(rlq_path):

    rlqimg = Image.open(rlq_path+"/"+rlq)
    i+=1
    if sw>w:
        sw=w
    if sh>h:
        sh=h

   #4. real lq image 저장하기 ------------------
    if step ==0 : 
        rlqimg.save(gen_path+"/trainA/"+str(j)+".jpg")
        print("["+str(i)+"] trainA saved")
    #val set
    if step ==1 : 
        rlqimg.save(gen_path+"/valA/"+str(j)+".jpg")
        print("["+str(i)+"] valA saved")
    #test set
    if step ==2 : 
        rlqimg.save(gen_path+"/testA/"+str(j)+".jpg")
        print("["+str(i)+"] testA saved")

    j+=1
    
    if (i==ltrainN):
        j=1 #초기화
        step+=1
        print("===[train set end]============================")
    if (i==(ltrainN+lvalN)):
        j=1
        step+=1
        print("===[val set end]============================")

print("smallest (width, height) =  (", sw, ", ", sh,")")

