import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# ----------------------------------------------------------------
# 1) 한세트 이미지 pair끼리 붙이기 - because pix2pix input(facade) looks like that
# 2) train, test, val set으로 나누고
# 3) 각각 file 명 1부터 시작하게
# 
# 참고: https://copycoding.tistory.com/159 
# 2022.01.04 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
image_size =500
rhq_path = "/home/guest1/ellen_data/UKB_quality_data/high_Q"
glq_path = "/home/guest1/ellen_data/UKB_quality_data/gen_lowQ_byHQ_202220103"
gen_path = "/home/guest1/ellen_data/UKB_quality_data/pix2pixinput_"+str(image_size)+"_202220104"
trainR, valR, testR= 6,2,2 # train : val: test 비율 쓰기


# 0-1. path없으면 path생성
if not os.path.isdir(gen_path+"/train"):
    os.makedirs(gen_path+"/train")
    print("train path generated!")
if not os.path.isdir(gen_path+"/val"):
    os.makedirs(gen_path+"/val")
    print("val path generated!")
if not os.path.isdir(gen_path+"/test"):
    os.makedirs(gen_path+"/test")
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
# 1. read hq import image ------------------
for rhq in os.listdir(rhq_path):

    #2. real hq image size256로 바꾸기  ------------------
    # rhqimg = cv.imread(rhq_path+"/"+rhq,1)
    rhqimg = Image.open(rhq_path+"/"+rhq)
    rhqimg_resize= rhqimg.resize((image_size,image_size),Image.LANCZOS)
    #3. gen lq image 찾기 &  붙이기  ------------------
    for glq in os.listdir(glq_path):
        if(glq==rhq): 
            # glqimg =cv.imread(glq_path+"/"+glq,1)
            glqimg =Image.open(glq_path+"/"+glq)
            glqimg_resize= glqimg.resize((image_size,image_size),Image.LANCZOS)

            glq_rhq = np.hstack((glqimg_resize,rhqimg_resize))
    i+=1

    print(glq_rhq.shape)
    glq_rhq_i = Image.fromarray(glq_rhq)
    #4. image저장  ------------------
    # #train set --> cv imwrite은 왜인지 안됨
    # if step ==0 : 
    #     cv.imwrite(gen_path+"/train/"+str(j)+".jpg", glq_rhq)
    #     print("saved")
    # #val set
    # if step ==1 : 
    #     cv.imwrite(gen_path+"/val/"+str(j)+".jpg", glq_rhq)
    # #test set
    # if step ==2 : 
    #     cv.imwrite(gen_path+"/test/"+str(j)+".jpg", glq_rhq)
    #train set
    if step ==0 : 
        glq_rhq_i.save(gen_path+"/train/"+str(j)+".jpg")
        print("["+str(i)+"] train saved")
    #val set
    if step ==1 : 
        glq_rhq_i.save(gen_path+"/val/"+str(j)+".jpg")
        print("["+str(i)+"] val saved")
    #test set
    if step ==2 : 
        glq_rhq_i.save(gen_path+"/test/"+str(j)+".jpg")
        print("["+str(i)+"] test saved")

    j+=1
    
    if (i==trainN):
        j=1 #초기화
        step+=1
        print("===[train set end]============================")
    if (i==(trainN+valN)):
        j=1
        step+=1
        print("===[val set end]============================")

