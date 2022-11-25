import os
from tqdm import tqdm
import shutil
path = "/home/guest1/ellen_code/pytorch-CycleGAN-and-pix2pix_ellen/checkpoints"

#주의 파일들 그냥 복원안되게 지워지니까 주의할 것 
# 1. pth파일 latest빼고
# 2. web에 올리는 image들 있는 폴더 지우기 


# for folder in os.listdir(path):
#     print(folder)
#     for files in tqdm(os.listdir(path+"/"+folder)):
#         # 1. pth latest 빼고 다 지움
#         if ("pth" in files) and ("latest" not in files):
#             os.remove(path+"/"+folder+"/"+files)
#     print("------------------------------------")
#     for files in tqdm(os.listdir(path+"/"+folder)):
#         # 2. web에 올리는 image들 있는 폴더 지우기 
#         if "web" in files:
#             shutil.rmtree(path+"/"+folder+"/"+files)
            

