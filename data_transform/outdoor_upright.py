# -----------------------------
# image rotate 90
# -----------------------------
# for nonhomogeneous outdoor data(1200 x 1600 size) from CVPRw
# outdoor data중 몇개가 90도 돌아가 있어서
# data from: https://competitions.codalab.org/competitions/22236#participate
#          : https://competitions.codalab.org/competitions/28032#participate
# reference: https://pythonexamples.org/python-pillow-rotate-image-90-180-270-degrees/
#
# made by: Ellen jieun oh
# date: 2022.07.14
# email: jieunoh@postech.ac.kr
# modified: 2022.07.14


import os
from PIL import Image
import cv2 as cv
import numpy as np

ori_path = "/home/guest1/ellen_data/outdoor_rename"
gen_path = "/home/guest1/ellen_data/outdoor_rename_upright"

folders = ['trainA', 'trainB', 'valA', 'valB', 'testA', 'testB']
rotated_list = [[1, 2, 5, 6, 11, 21], [12, 17, 18], [], [
    1, 2, 5, 6, 11], [4, 8, 9, 10, 12], [4, 8, 9, 10, 12]]  # 돌아간 이름 폴더 순으로

# ---------------------------------
# 1. setting
# ---------------------------------
# folder가 없으면, 생성 ---------------------------------
if not os.path.isdir(gen_path):
    for f in folders:
        os.makedirs(gen_path+"/"+f)
        print(f + " generated!")
print("Path generated END----------------------------------------------------------------")


# ---------------------------------
# 2. image rotate
# ---------------------------------
# [1] 폴더별로 열기-----------------
for i, f in enumerate(folders):
    rotation_check = rotated_list[i]
    count = 0
    # [2] 폴더 안 열기-----------------
    for ith, img in enumerate(sorted(os.listdir(ori_path+"/"+f))):
        if ith in rotation_check:
            im = Image.open(ori_path+"/"+f+"/"+img)
            imr = im.rotate(-90, expand = True)
            imr.save(gen_path+"/"+f+"/"+img)
            count += 1
        else:
            im =Image.open(ori_path+"/"+f+"/"+img)
            im.save(gen_path+"/"+f+"/"+img)
    print("["+f+"] "+str(count)+"images filped----")
print("Image cropping END----------------------------------------------------------------")