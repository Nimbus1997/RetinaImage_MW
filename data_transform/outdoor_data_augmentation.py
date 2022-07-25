# -----------------------------
# imate preprocessing - crop(square), resize
# image augmentation - flip, rotation 
# -----------------------------
# for nonhomogeneous outdoor data(1200 x 1600 size) from CVPRw
# outdoor data중 몇개가 90도 돌아가 있어서
# data from: https://competitions.codalab.org/competitions/22236#participate
#          : https://competitions.codalab.org/competitions/28032#participate
# reference: https://pythonexamples.org/python-pillow-rotate-image-90-180-270-degrees/
#
#   1. crop *8: 400날리는데 50단위로 잘라서 (1200,1600),-> (1200,1200) ".crop" https://appia.tistory.com/367 
#   2. flip *2: horizontal ".transpose" https://appia.tistory.com/368
#   3. resize *1:(1200,1200) -> (512,512) ".resize" https://ponyozzang.tistory.com/600  
#   4. rotation *7: -15, -10,-5, 0, 5, 10 ,15 ".rotate" https://pythonexamples.org/python-pillow-rotate-image-90-180-270-degrees/
#
# made by: Ellen jieun oh
# date: 2022.07.14
# email: jieunoh@postech.ac.kr
# modified: 2022.07.14


import os
from PIL import Image
import cv2 as cv
import numpy as np

ori_path = "/home/guest1/ellen_data/outdoor_rename_upright"
gen_path = "/home/guest1/ellen_data/outdoor_rename_512_augmentation"

gen_list =['1crop', '2flip', '3final']
folders = ['trainA', 'trainB', 'valA', 'valB', 'testA', 'testB']
# rotated_list = [[1, 2, 5, 6, 11, 21], [12, 17, 18], [], [
#     1, 2, 5, 6, 11], [4, 8, 9, 10, 12], [4, 8, 9, 10, 12]]  # 돌아간 이름 폴더 순으로


# ---------------------------------
# 1. setting
# ---------------------------------
# folder가 없으면, 생성 ----------------
for gen_f in gen_list:
    if not os.path.isdir(gen_path+gen_f):
        for f in folders:
            os.makedirs(gen_path+gen_f+"/"+f)
    print(gen_f + " generated!")
print("Path generated END----------------------------------------------------------------")

# Crop ----------------
space50 = list(range(0,1600-1200+1, 50))

# Rotation ----------------
rotation5 = list(range(-15,16,5))

# 확인 -----------------
print("-----------------------------")
print("ori_path: ", ori_path)
print("gen_path: ", gen_path)
print("crop space:",space50)
print("rotation angle: ", rotation5)
print("train, val, test: ", len(space50)*2*len(rotation5), len(space50)*2, len(space50)*2)
print("-----------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>")



# ---------------------------------
# 2. image augmentation
# ---------------------------------
step = 0

# # [1] CROP -----------------
# for f in folders:
#     for img in sorted(os.listdir(ori_path+"/"+f)):
#         im =Image.open(ori_path+"/"+f+"/"+img)
#         i=0
#         if im.size[0]==1200: # 세로로 긴 것
#             for space in space50:
#                 croppedim = im.crop((0, space, 1200, 1200+space))
#                 croppedim.save(gen_path+gen_list[step]+"/"+f+"/"+img.split(".")[0]+"_"+str(i)+".jpg")
#                 i+=1

#         else: # 가로로 긴 것
#             for space in space50:
#                 croppedim = im.crop((space, 0, 1200+space, 1200))
#                 croppedim.save(gen_path+gen_list[step]+"/"+f+"/"+img.split(".")[0]+"_"+str(i)+".jpg")
#                 i+=1
# print("Image cropping END----------------------------------------------------------------")
# input(">>>>>>>check the 1crop folder and press 'enter' to proceed")
step += 1

# [2] FILP-----------------
# for f in folders:
#     for img in sorted(os.listdir(gen_path+gen_list[step-1]+"/"+f)):
#         im =Image.open(gen_path+gen_list[step-1]+"/"+f+"/"+img)
#         filpim = im.transpose(Image.FLIP_LEFT_RIGHT)
#         filpim.save(gen_path+gen_list[step]+"/"+f+"/"+img.split(".")[0]+"_"+str(1)+".jpg")
#         im.save(gen_path+gen_list[step]+"/"+f+"/"+img.split(".")[0]+"_"+str(0)+".jpg")
# print("Image filpping END----------------------------------------------------------------")
# input(">>>>>>>check the 2filp folder and press 'enter' to proceed")
step += 1


# [3] FILP-----------------
for f in folders:
    for img in sorted(os.listdir(gen_path+gen_list[step-1]+"/"+f)):
        im =Image.open(gen_path+gen_list[step-1]+"/"+f+"/"+img)
        if 'train' in f: 
            for i, angle in enumerate(rotation5):
                rotateim = im.rotate(angle)
                rotateim.save(gen_path+gen_list[step]+"/"+f+"/"+img.split(".")[0]+"_"+str(i)+".jpg")
        else:
            im.save(gen_path+gen_list[step]+"/"+f+"/"+img.split(".")[0]+"_"+str(0)+".jpg")
step += 1
print("Image Rotation END----------------------------------------------------------------")
print(">>>>>>>>>>>>>>>>END<<<<<<<<<<<<<<<<")
