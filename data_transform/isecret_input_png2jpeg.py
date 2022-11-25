import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

#___________
# READ ME-----------------------------------------------------
# 1)png file -> jpeg file
# ----------------------------------------------------------------

# 0. augment 설정 ------------------

path = "/home/guest1/ellen_data/input_eyeq_20221107_512_n1000/eyeq"
imageformat="png" #원본 이미지 포멧
newformat ="jpeg"

folders =["train", "val", "test"]
folders2 = ['crop_good', 'crop_usable','degrade_good','gt_cropgood_resize']
for f in folders:
    for ff in folders2:
        print("now in>>>", f, ff)
        for imgg in tqdm(os.listdir(path+'/'+f+'/'+ff)):
            if imageformat in imgg:
                oimg = Image.open(path+'/'+f+'/'+ff+"/"+imgg)
                neeimg= imgg.split(".")[:-1]
                oimg.save(path+'/'+f+'/'+ff+"/"+neeimg+".",newformat)
                os.remove(path+'/'+f+'/'+ff+"/"+imgg)