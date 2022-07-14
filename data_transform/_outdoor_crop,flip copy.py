# -----------------------------
# (1) image 512 crop & (2)flip & _00(f).jpg 로 생성됨.
# test * val : 한 이미지 당 12개 생성
# train : 한 이미지당 *40(crop) *2 (flip)
# -----------------------------
# for nonhomogeneous outdoor data(1200 x 1600 size) from CVPRw
# data from: https://competitions.codalab.org/competitions/22236#participate
#          : https://competitions.codalab.org/competitions/28032#participate
# reference: https://webnautes.tistory.com/1653
#
# made by: Ellen jieun oh 
# date: 2022.07.10
# email: jieunoh@postech.ac.kr
# modified: 2022.07.14


import os
import cv2 as cv
import numpy as np

ori_path = "/home/guest1/ellen_data/outdoor_rename"
gen_path = "/home/guest1/ellen_data/outdoor_rename_512"

folders = ['trainA','trainB', 'valA', 'valB', 'testA','testB']
rotated_list = [[1,2,5,6,11,21],[12,17,18], [],[1,2,5,6,11], [4,8,9,10,12],[4,8,9,10,12]] #돌아간 이름 폴더 순으로 

# ---------------------------------
# 1. setting 
# ---------------------------------
# [Def]CROP: center 값 주어면, size *size로 crop ---------------------------------
def center_crop(img:np.array,x:int,y:int, size:int) -> np.array:
    ss = size//2
    # size == 512
    crop_img=img[x-ss:x+ss, y-ss:y+ss,:]
    return crop_img

# crop 어떻게 할지 center point - x, y ---------------------------------
size = 512
height = 1200
width = 1600

# test& val는 조금만 자름
tvx=[256, 344 ,688]# x=[size//2, (size//2+(1200-size//2))//2 ,1200-size//2]
tvy=[256,618,982,1344]# y=[size//2, 사이//3 해서 등간격 ,1600-size//2]
# train은 자잘하게 자름 
x=list(range(256,688,100)) #  5개
y=list(range(256,1344,150)) # 8개 


# folder가 없으면, 생성 ---------------------------------
if not os.path.isdir(gen_path):
    for f in folders:
        os.makedirs(gen_path+"/"+f)
        print(f+ " generated!")
print("Path generated END----------------------------------------------------------------")


# ---------------------------------
# 2. image rotate, crop, flip
# ---------------------------------
# [1] 폴더별로 열기-----------------
for ii,f in enumerate(folders):
    rotation_check = rotated_list[ii]
    # [2] 폴더별로 열기-----------------
    if "train" in f:
        for ith, img in enumerate(sorted(os.listdir(ori_path+"/"+f))):
            imgg = cv.imread(ori_path+"/"+f+"/"+img)
            roated= False
            i = 0

            # # image가 돌아가 있는 경우는 사진을 돌리고, 범위도 바꿔줘야함
            # if ith in rotation_check: 
            #     temp = x
            #     x=y
            #     y=temp
                
            
            for xx in x:
                for yy in y:
                    
                    imgg_croped = center_crop(imgg, xx,yy, size)
                    cv.imwrite(gen_path+"/"+f+"/"+img.split(".")[0]+"_"+str(i).zfill(2)+".png", imgg_croped)

                    i+=1
        print(f+" croped & filped----")

    else:
        for img in sorted(os.listdir(ori_path+"/"+f)):
            imgg = cv.imread(ori_path+"/"+f+"/"+img)
            i = 0
            for xx in tvx:
                for yy in tvy:
                    imgg_croped = center_crop(imgg, xx,yy, size)
                    cv.imwrite(gen_path+"/"+f+"/"+img.split(".")[0]+"_"+str(i).zfill(2)+".png", imgg_croped)
                    i+=1
        print(f+" croped----")
print("Image cropping END----------------------------------------------------------------")

