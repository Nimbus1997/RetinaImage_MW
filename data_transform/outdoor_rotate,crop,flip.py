# -----------------------------
# (0) image 90 도 돌아간 것 돌리고, crop 시작
# (1) image crop & (2)flip & _00(f).jpg 로 생성됨.
# test * val : 한 이미지 당 12개 생성
# train : 한 이미지당 *45(crop) *2 (flip)
# -----------------------------
# for nonhomogeneous outdoor data(1200 x 1600 size) from CVPRw
# data from: https://competitions.codalab.org/competitions/22236#participate
#          : https://competitions.codalab.org/competitions/28032#participate
# reference: https://webnautes.tistory.com/1653
#
# made by: Ellen jieun oh 
# date: 2022.07.10
# email: jieunoh@postech.ac.kr
# modified: 


import os
import cv2 as cv
import numpy as np

ori_path = "/home/guest1/ellen_data/outdoor_rename"
gen_path = "/home/guest1/ellen_data/outdoor_rename_512"

folders = ['trainA','trainB', 'valA', 'valB', 'testA','testB']

def center_crop(img:np.array,x:int,y:int, size:int) -> np.array:
    ss = size//2
    # size == 512
    crop_img=img[x-ss:x+ss, y-ss:y+ss,:]
    return crop_img

# crop 어떻게 할지 center point - x, y ---------------------------------
size = 512
x_width=1200
y_width= 1600

tvx=[256, 344 ,688]
# x=[size//2, (size//2+(1200-size//2))//2 ,1200-size//2]
tvy=[256,618,982,1344]
# y=[size//2, 사이//3 해서 등간격 ,1600-size//2]

x=list(range(256,688,150))
y=list(range(256,1344,150))

# folder가 없으면, 생성 ---------------------------------
if not os.path.isdir(gen_path):
    for f in folders:
        os.makedirs(gen_path+"/"+f)
        print(f+ " generated!")
print("Path generated END----------------------------------------------------------------")


# 이미지 crop  & 저장 --------------------------------- 
for f in folders:
    if "train" in f:
        for img in sorted(os.listdir(ori_path+"/"+f)):
            imgg = cv.imread(ori_path+"/"+f+"/"+img)
            i = 0
            for xx in x:
                for yy in y:
                    imgg_croped = center_crop(imgg, xx,yy, size)
                    cv.imwrite(gen_path+"/"+f+"/"+img.split(".")[0]+"_"+str(i).zfill(2)+".jpg", imgg_croped)

                    i+=1
        print(f+" croped & filped----")

    else:
        for img in sorted(os.listdir(ori_path+"/"+f)):
            imgg = cv.imread(ori_path+"/"+f+"/"+img)
            i = 0
            for xx in tvx:
                for yy in tvy:
                    imgg_croped = center_crop(imgg, xx,yy, size)
                    cv.imwrite(gen_path+"/"+f+"/"+img.split(".")[0]+"_"+str(i).zfill(2)+".jpg", imgg_croped)
                    i+=1
        print(f+" croped----")
print("Image cropping END----------------------------------------------------------------")

