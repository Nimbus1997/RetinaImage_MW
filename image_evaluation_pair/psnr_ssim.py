#-----------------------------------------------
## Paired data quality check : PSNR, SSIM
#-----------------------------------------------
# 참고 개념: https://m.blog.naver.com/mincheol9166/221771426327
# 참고 코드 : scikit : https://scikit-image.org/docs/stable/api/skimage.metrics.html
# psnr: skimage.metrics.peak_signal_noise_ratio
# ssim: skimage.metrics.structural)similarity

# 스스로 psnr, ssim 코드 짜고 싶을 때: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

# 작성: 오지은 (jieunoh@postech.ac.kr)
# 날짜: 2022.07.11
#-----------------

import skimage.metrics
import os
import cv2 as cv
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# 0.돌릴때 마다 설정------------------------------
path = ""
date = "0710"

# 0. 설정 no need to change this------------------------------
excel_path = "/home/guest1/ellen_code/RetinaImage_MW/image_evaluation_pair/pair_result/"
excel_name = "PAIR_"+path.split("/")[-4]+"_"+date+".csv"
plot_name = "PAIR_"+path.split("/")[-4]+"_"+date+".png"

print("----------------------------")
print("original data path:")
print("gen path:")
print("excel_name:")
print("plot_name:")
print("----------------------------")
input("위의 값 확인후 enter를 눌러서 진행, 멈추려면 ctrl+c >>>")

types = ['real_A', 'real_B', 'fake_A', 'fake_B', 'rec_A', 'rec_B']

# 1. excel 기본 설정 ------------------------------
df = pd.DataFrame(columns=['image_name', 'psnr_lh',
                  'psnr_hh', 'ssim_lh', 'ssim_hh'])

# 2. psnr, ssim ------------------------------
#    def
def psnr_ssim_cal(real_h: np.array, real_l: np.array, gen_h: np.array) -> float:
    psnr_lh = skimage.metrics.peak_signal_noise_ratio(real_h, real_l)
    psnr_hh = skimage.metrics.peak_signal_noise_ratio(real_h, gen_h)
    ssim_lh = skimage.metrics.structural_similarity(real_h, real_l)
    ssim_hh = skimage.metrics.structural_similarity(real_h, gen_h)
    return psnr_lh, psnr_hh, ssim_lh, ssim_hh


# 2. 실제로 돌리기 ------------------------------
count = 0
for image in sorted(os.listdir(path)):
    # image name example: 000_00_real_A.jpg
    if 'real_B' in image:
        real_high = image
        name = "_".join(image.split("_")[0:2])
        for img in sorted(os.listdir(path)):
            if name in img and 'real_A' in img:
                real_low = img
                break
        for img in sorted(os.listdir(path)):
            if name in img and 'fake_B' in img:
                gen_high = img
                break
        real_highi = cv.imread(path+"/"+real_high)
        real_lowi = cv.imread(path+"/"+real_low)
        gen_highi = cv.imread(path+"/"+gen_high)

        psnr_lh, psnr_hh, ssim_lh, ssim_hh = psnr_ssim_cal(
            real_highi, real_lowi, gen_highi)

        # dataframe에 저장
        df.loc[count] = [name, psnr_lh, psnr_hh, ssim_lh, ssim_hh]
        count +=1


# 3. excel 저장 ------------------------------
df.to_csv(excel_path + excel_name)
print("-----------------------------")
print("cvs saved!!! : "+excel_name)
print("path: "+excel_path+excel_name)

# 4. PSNR, SSIM 평균값 ------------------------------
roundd=2
psnr_lhm=str(round(df[psnr_lh].mean(),roundd))
psnr_lhs=str(round(df[psnr_lh].std(),roundd))
psnr_hhm=str(round(df[psnr_hh].mean(),roundd))
psnr_hhs=str(round(df[psnr_hh].std(),roundd))

ssim_lhm=str(round(df[ssim_lh].mean(),roundd))
ssim_lhs=str(round(df[ssim_lh].std(),roundd))
ssim_hhm=str(round(df[ssim_hh].mean(),roundd))
ssim_hhs=str(round(df[ssim_hh].std(),roundd))
information = "lh& hh (mean,std)\n PSNR:("+psnr_lhm+", "+psnr_lhs+"), (" +psnr_hhm+" ,"+psnr_hhs+")\n SSIM:("+ssim_lhm+", "+ssim_lhs+"), (" +ssim_hhm+" ,"+ssim_hhs+")"
# 4. 그래프 그리기 ------------------------------

plt.subplot(2,1,1)
sns.displot(df['psnr_lh'], color='red', label="psnr_lh")
sns.displot(df['psnr_hh'], color='steelblue', label='psnr_hh')
plt.legend(title="type")
plt.gca().set_title("PSNR")

plt.subplot(2,1,2)
sns.displot(df['ssim_lh'], color='red', label="ssim_lh")
sns.displot(df['ssim_hh'], color='steelblue', label='ssim_hh')
plt.legend(title="type")
plt.gca().set_title("SSIM")

plt.suptitle(information)
plt.tight_layout()
plt.show()
plt.savefig(excel_path+plot_name)


# 5. 그래프 저장하기 ------------------------------
print("plot saved!!! : "+plot_name)
print("path: "+excel_path+plot_name)
print("-----------------------------")
