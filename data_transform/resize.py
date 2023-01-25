import os
from PIL import Image
import cv2 as cv
from torchvision.transforms import functional as F


# ----------------------------------------------------------------
# image resize - because pix2pix input size is originally 256 
# 2022.01.03 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
ori_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_hq"
gen_path = "/root/jieunoh/ellen_data/isecret_eyeq_total_hq_512" # change this
i=0
resize=512

# 0-1. path없으면 path생성 ------------------
if not os.path.isdir(gen_path):
    os.makedirs(gen_path)
    print("path generated!")

# 1. import image ------------------
for folder in os.listdir(ori_path): # 폴더안에 6개 폴더 있으니까 각각
    print("in ["+folder+"] now >>>>>>>>>>>>>>>>>>>>>>>")
    if not os.path.isdir(gen_path+"/"+folder):
        os.makedirs(gen_path+"/"+folder)

    for file in os.listdir(ori_path+"/"+folder):
        img = Image.open(ori_path+"/"+folder+"/"+file)
        # img_resize= img.resize((resize,resize),Image.LANCZOS)
        img_resize= F.resize(img,(resize,resize),  Image.BICUBIC)
        img_resize.save(gen_path+"/"+folder+"/"+file)

