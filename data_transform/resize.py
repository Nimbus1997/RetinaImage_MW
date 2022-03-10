import os
from PIL import Image
from torch import ge


# ----------------------------------------------------------------
# image resize - because pix2pix input size is originally 256 
# 2022.01.03 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
ori_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220220"
gen_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220221"
i=0

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
        img_resize= img.resize((512,512),Image.LANCZOS)
        img_resize.save(gen_path+"/"+folder+"/"+file)

