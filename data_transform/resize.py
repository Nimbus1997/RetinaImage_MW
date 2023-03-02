import os
from PIL import Image
import cv2 as cv
from torchvision.transforms import functional as F
from tqdm import tqdm


# ----------------------------------------------------------------
# image resize - because pix2pix input size is originally 256 
# 2022.01.03 jieunoh@postech.ac.kr
# (edit)2023.01.25  resize by torchvision.transfroms.functional.resize
# ----------------------------------------------------------------

# 0. augment 설정 ------------------
ori_path = "/root/jieunoh/ellen_data/0_isecret_input_reject"
gen_path = "/root/jieunoh/ellen_data/0_isecret_input_reject_256" # change this
i=0
resize=256

# 0-1. path없으면 path생성 ------------------
if not os.path.isdir(gen_path):
    os.makedirs(gen_path)
    print("path generated!")

# >> path안에 폴더가 들어있는 경우 <cycle GAN>
# # 1. import image ------------------
# for folder in os.listdir(ori_path): # 폴더안에 6개 폴더 있으니까 각각
#     print("in ["+folder+"] now >>>>>>>>>>>>>>>>>>>>>>>")
#     if not os.path.isdir(gen_path+"/"+folder):
#         os.makedirs(gen_path+"/"+folder)

#     for file in os.listdir(ori_path+"/"+folder):
#         img = Image.open(ori_path+"/"+folder+"/"+file)
#         # img_resize= img.resize((resize,resize),Image.LANCZOS)
#         img_resize= F.resize(img,(resize,resize),  Image.BICUBIC)
#         img_resize.save(gen_path+"/"+folder+"/"+file)

# # >> path안에 바로 이미지가 들어있는 경우
# for file in tqdm(sorted(os.listdir(ori_path))):
#     img = Image.open(ori_path+"/"+file)
#     # img_resize= img.resize((resize,resize),Image.LANCZOS)
#     img_resize= F.resize(img,(resize,resize),  Image.BICUBIC)
#     img_resize.save(gen_path+"/"+file)


# >> path안에 폴더가 들어있는 경우 <ISECRET>
# # 1. import image ------------------

group1 = ["train","val","test"]  # outer directory
group2 = ["crop_good","crop_usable","degrade_good"]  # inner directories 

# 0-1. path없으면 path생성 ------------------------------------------------------------------------------------------
print("--------------------------")
print(gen_path)

for folder in os.listdir(ori_path): 
    for g1 in group1:
        for g2 in group2:
            if not os.path.isdir(gen_path+"/"+folder+"/eyeq/"+g1+"/"+g2):
                print("made folder: ",gen_path+"/"+folder+"/eyeq/"+g1+"/"+g2)
                os.makedirs(gen_path+"/"+folder+"/eyeq/"+g1+"/"+g2)
            
            for files in tqdm(os.listdir(ori_path+"/"+folder+"/eyeq/"+g1+"/"+g2)):
                img = Image.open(ori_path+"/"+folder+"/eyeq/"+g1+"/"+g2+"/"+files)
                # img_resize= img.resize((resize,resize),Image.LANCZOS)
                img_resize= F.resize(img,(resize,resize),  Image.BICUBIC)
                img_resize.save(gen_path+"/"+folder+"/eyeq/"+g1+"/"+g2+"/"+files)



