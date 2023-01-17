#-----------------------------------------------
## paired data quality check : PSNR, SSIM
#-----------------------------------------------
# real low, gen high 를 pair로 계산

# 참고 개념: https://m.blog.naver.com/mincheol9166/221771426327
# 참고 코드 : scikit : https://scikit-image.org/docs/stable/api/skimage.metrics.html
# psnr: skimage.metrics.peak_signal_noise_ratio
# ssim: skimage.metrics.structural)similarity

# 스스로 psnr, ssim 코드 짜고 싶을 때: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

# 작성: 오지은 (jieunoh@postech.ac.kr)
# 날짜: 2022.01.03
#-----------------

import skimage.metrics
import os
import cv2 as cv
import numpy as np
import pdb
from tqdm import tqdm

# 0.돌릴때 마다 설정------------------------------
path = "/home/guest1/ellen_code/pytorch-CycleGAN-and-pix2pix_ellen/results/compare_unet_3_1213/test_185/images"

# 0. 설정 no need to change this------------------------------


print("----------------------------")
print("original data path:", path)
print("----------------------------")
input("위의 값 확인후 enter를 눌러서 진행, 멈추려면 ctrl+c >>>")


# 1. psnr, ssim ------------------------------
#    def
def psnr_ssim_cal(real_l: np.array, gen_h: np.array) -> float:
    psnr_ch=[]
    ssim_ch=[]

    for i in range(3):      
        psnr_ch.append(skimage.metrics.peak_signal_noise_ratio(gen_h[:,:,i], real_l[:,:,i]))
        ssim_ch.append(skimage.metrics.structural_similarity(gen_h[:,:,i], real_l[:,:,i]))

    return np.mean(psnr_ch), np.mean(ssim_ch)

psnr_list=[]
ssim_list=[]
gen_h_num ="none_gen"
real_l_num ="none_real"

# 2. 실제로 돌리기 ------------------------------
total =0 
for image in tqdm(sorted(os.listdir(path))):
    # image name example: 0000_real_A.jpg
    num= image.split("_")[0] # 0000
    if 'fake_B' in image and num in image:
        gen_h=cv.imread(path+"/"+image)
        gen_h_num= num
    if 'real_B' in image and num in image:
        real_l = cv.imread(path+"/"+image)
        real_l_num=num
    if gen_h_num==real_l_num and 'real_B' in image :
        total +=1
        psnr_rlgh, ssim_rlgh = psnr_ssim_cal(
            gen_h,real_l)
        # print("num:", gen_h_num, " , psnr: ", psnr_rlgh, " , ssim: ", ssim_rlgh)
        psnr_list.append(psnr_rlgh)
        ssim_list.append(ssim_rlgh)
psnr = np.mean(psnr_list)
ssim = np.mean(ssim_list)

print("-------------------------------")
print(path)
print(total)
print("PSRN: ", psnr)
print("SSIM: ", ssim)
print("-------------------------------")        

