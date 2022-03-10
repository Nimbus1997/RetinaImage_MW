import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# ----------------------------------------------------------------
# 1) train-A,B, test-A,B, val-A,B set으로 나누고
# 2) 각각 file 명 1부터 시작하게
# 3) GLQ로 B SET만들고 A는 RHQ로 구성
# --> pix2pix 과 다른것은 원래 크기대로. --> 원래 크기대로하면 크기 안맞아서 문제 될 수 있는데, 문제되면 그때 고치겠음
# 
# 참고: https://copycoding.tistory.com/159 
# 2022.01.10 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
rhq_path = "/home/guest1/ellen_data/UKB_quality_data/high_Q"
glq_path = "/home/guest1/ellen_data/UKB_quality_data/gen_lowQ_byHQ_202220103" #원본크기
gen_path = "/home/guest1/ellen_data/UKB_quality_data/cycleGANinput_202220110"
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


total_dataset = len(os.listdir(rhq_path))
trainN = int(float(total_dataset)/10.*float(trainR))
valN = int(float(total_dataset)/10.*float(valR)) 
testN = total_dataset - trainN-valN

print("ratio of dataset (train : val : test) : (", trainR, ": ", valR , ": ", testR ,")")
print("# of dataset in (train : val : test) : (", trainN, ", ", valN , ", ", testN ,")")

i=0
step=0
j=1
# 1. real hq import image ------------------
for rhq in os.listdir(rhq_path):

    #2. real hq image 불러오기  ------------------
    # rhqimg = cv.imread(rhq_path+"/"+rhq,1)
    rhqimg = Image.open(rhq_path+"/"+rhq)
    # rhqimg_resize= rhqimg.resize((256,256),Image.LANCZOS)
    #3. gen lq image 찾기 &  붙이기  ------------------
    for glq in os.listdir(glq_path):
        if(glq==rhq): 
            # glqimg =cv.imread(glq_path+"/"+glq,1)
            glqimg =Image.open(glq_path+"/"+glq)

            glq_rhq = np.hstack((glqimg,rhqimg))
    i+=1

    print(glq_rhq.shape)
    glq_rhq_i = Image.fromarray(glq_rhq)
    
    if step ==0 : 
        glq_rhq_i.save(gen_path+"/train/"+str(j)+".jpg")
        print("["+str(i)+"] train saved")
        glqimg.save(gen_path+"/trainA/"+str(j)+".jpg")
        print("["+str(i)+"] trainA saved")
        rhqimg.save(gen_path+"/trainB/"+str(j)+".jpg")
        print("["+str(i)+"] trainB saved")
    #val set
    if step ==1 : 
        glq_rhq_i.save(gen_path+"/val/"+str(j)+".jpg")
        print("["+str(i)+"] val saved")
        glqimg.save(gen_path+"/valA/"+str(j)+".jpg")
        print("["+str(i)+"] valA saved")
        rhqimg.save(gen_path+"/valB/"+str(j)+".jpg")
        print("["+str(i)+"] valB saved")
    #test set
    if step ==2 : 
        glq_rhq_i.save(gen_path+"/test/"+str(j)+".jpg")
        print("["+str(i)+"] test saved")
        glqimg.save(gen_path+"/testA/"+str(j)+".jpg")
        print("["+str(i)+"] testA saved")
        rhqimg.save(gen_path+"/testB/"+str(j)+".jpg")
        print("["+str(i)+"] testB saved")

    j+=1
    
    if (i==trainN):
        j=1 #초기화
        step+=1
        print("===[train set end]============================")
    if (i==(trainN+valN)):
        j=1
        step+=1
        print("===[val set end]============================")

