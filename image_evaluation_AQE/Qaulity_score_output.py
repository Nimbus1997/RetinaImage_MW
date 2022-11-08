# ------------------------------------------------
# read me
# > for output data
# > shaprness.py와 illumination.py의 함수를 이용해서 최종 quality score를 구해서 excel에 작성
# > excel의 column: [1] type: RHQ(real_B) or RLQ(real_A 중 not h) or GHQ(fake_B) ,fake_A, rec_A, rec_B 총 6개
#                   [3] ImageName,
#                   [4] sharpness,
#                   [5] illumination,
#                   [6] quality_score,
#                   [7] category(1: verygood, 2:good, 3: acceptable, 4 not acceptable)
# > excel name: AQE_data폴더이름_date
# AQE: by H.Bartling ‘Automated Quality Evaluation of Digital Fundus Photography’ 2009 Acta Ophthalmologica (impact factor: 3.761, # cite: 68)
# > 2022.06.27 jieunoh@postech.ac.kr
# >     .06.30 Quality score = S * I 로 수정함. +로 오타 나있었음 
# ------------------------------------------------
from illumination import illumination
from sharpness import sharpness
import numpy as np
import os
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# 0. 돌릴때 마다 설정 ---------------------------
image_size = 512
patch_size = 64
path = "/home/guest1/ellen_code/pytorch-CycleGAN-and-pix2pix_ellen/results/ellen_dwt_uresnet1_3_512n1000_0815_b4/test_latest/images/"
# date = '0704'

# 0. 설정 ---------------------------
# no need to change this
excel_path = "/home/guest1/ellen_code/RetinaImage_MW/image_evaluation_AQE/AQE_result/output/"
excel_name = "AQE_"+path.split("/")[-4]+"_p" + str(patch_size)+".csv"
plot_name = "AQE_"+path.split("/")[-4]+"_p" + str(patch_size)+".png"

num = image_size/patch_size
print("-------------------------------")
print("image size: ", image_size)
print("patch_size: ", patch_size)
print("orignal_data: ", path.split("/")[-4])

print("input path:", path)
print("excel_name: ", excel_name)
print("plot_name: ", plot_name)
print("-------------------------------")
input("위의 값 확인후 enter 눌러서 진행 >>>")  # 확인후 넘어가게 입력되면 넘어가게

types = ['real_A', 'real_B', 'fake_A', 'fake_B', 'rec_A', 'rec_B']

# df = pd.read_csv(excel_path+excel_name)

# 0. def category ---------------------------
def category(quality_score: int) -> int:
    if quality_score > 10:
        #very good
        return 1
    elif quality_score > 6:
        #good
        return 2
    elif quality_score > 2:
        #acceptable
        return 3
    else:
        #un acceptable
        return 4


# 1. excel 기본 설정 ---------------------------
df = pd.DataFrame(columns=['type', 'image_name', 'sharpness',
                  'illumination', 'quality_score', 'category'])
count = 0

# 2. RHQ -input
for typei in types:
    for image in sorted(os.listdir(path)):
        if (typei in image) and ("h" not in image):
            img = cv.imread(path+image, 0)
            sharpnessi = sharpness(img, num)
            illuminationi = illumination(img, num)
            quality_scorei = sharpnessi*illuminationi
            categoryi = category(quality_scorei)

            row = [typei, image, sharpnessi,
                   illuminationi, quality_scorei, categoryi]
            df.loc[count] = row
            count += 1
    print(">>>>END: "+typei)

# 5-0. Sharpness, Iluumination, Quality score mean& std --------------
# ra: real A, rb: realB, rca: rec_A
roundd = 2
ra_sm = str(round(df[df["type"] == 'real_A']["sharpness"].mean(), roundd))
ra_ss = str(round(df[df["type"] == 'real_A']["sharpness"].std(), roundd))
rb_sm = str(round(df[df["type"] == 'real_B']["sharpness"].mean(), roundd))
rb_ss = str(round(df[df["type"] == 'real_Bh']["sharpness"].std(), roundd))
fb_sm = str(round(df[df["type"] == 'fake_B']["sharpness"].mean(), roundd))
fb_ss = str(round(df[df["type"] == 'fake_B']["sharpness"].std(), roundd))

ra_im = str(round(df[df["type"] == 'real_A']["illumination"].mean(), roundd))
ra_is = str(round(df[df["type"] == 'real_A']["illumination"].std(), roundd))
rb_im = str(round(df[df["type"] == 'real_B']["illumination"].mean(), roundd))
rb_is = str(round(df[df["type"] == 'real_Bh']["illumination"].std(), roundd))
fb_im = str(round(df[df["type"] == 'fake_B']["illumination"].mean(), roundd))
fb_is = str(round(df[df["type"] == 'fake_B']["illumination"].std(), roundd))

ra_qm = str(round(df[df["type"] == 'real_A']["quality_score"].mean(), roundd))
ra_qs = str(round(df[df["type"] == 'real_A']["quality_score"].std(), roundd))
rb_qm = str(round(df[df["type"] == 'real_B']["quality_score"].mean(), roundd))
rb_qs = str(round(df[df["type"] == 'real_Bh']["quality_score"].std(), roundd))
fb_qm = str(round(df[df["type"] == 'fake_B']["quality_score"].mean(), roundd))
fb_qs = str(round(df[df["type"] == 'fake_B']["quality_score"].std(), roundd))


information = "realA-realB-fakeB: (mean,std): line1.s, line2.i ,line3.q \n (" +\
    ra_sm+", "+ra_ss+") ("+rb_sm+", "+rb_ss+")("+fb_sm+", "+fb_ss + ")\n ("\
    + ra_im+", "+ra_is+") ("+rb_im+", "+rb_is+")("+fb_im+", " + fb_is+")\n ("\
    + ra_qm+", "+ra_qs+") ("+rb_qm+", "+rb_qs+") ("+fb_qm+", "+fb_qs+")"


# 5. plot -----------------------------
# sharpness

plt.subplot(3, 1, 1)
sns.distplot(df[df["type"] == 'real_A']["sharpness"], color='red', label='real_A')
sns.distplot(df[df["type"] == 'real_B']["sharpness"],
             color='skyblue', label='real_B ')
sns.distplot(df[df["type"] == 'fake_B']["sharpness"], color='steelblue', label='fake_B')
plt.legend(title="type")
# plt.gca().set_title("sharpness")


# illumination
plt.subplot(3, 1, 2)
sns.distplot(df[df["type"] == 'real_A']["illumination"], color='red', label='real_A')
sns.distplot(df[df["type"] == 'real_B']["illumination"],
             color='skyblue', label='real_B ')
sns.distplot(df[df["type"] == 'fake_B']["illumination"], color='steelblue', label='fake_B')
# plt.gca().set_title("illumination")


# quality score
plt.subplot(3, 1, 3)
sns.distplot(df[df["type"] == 'real_A']["quality_score"], color='red', label='real_A')
sns.distplot(df[df["type"] == 'real_B']["quality_score"],
             color='skyblue', label='real_B ')
sns.distplot(df[df["type"] == 'fake_B']["quality_score"], color='steelblue', label='fake_B')
# plt.gca().set_title("quality_score")

plt.suptitle(information)
plt.tight_layout()
plt.show()
plt.savefig(excel_path+plot_name)

# 6. save it to csv and plot  ---------------------------
df.to_csv(excel_path + excel_name)
print("-----------------------------")
print("cvs saved!!! : "+excel_name)
print("path: "+excel_path+excel_name)
print("plot saved!!! : "+plot_name)
print("path: "+excel_path+plot_name)
print("-----------------------------")
