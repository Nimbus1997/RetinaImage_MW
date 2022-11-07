import nntplib
import numbers
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

#___________
# READ ME-----------------------------------------------------
# 새로운 데이터 주어졌을 때 원래 데이터와 합치는 코드
# 이름이 겹치는 데이터가 있으면 그것은 빼고 옮김
#
# 참고: https://www.delftstack.com/ko/howto/python/python-move-file/
# 2022.02.18 jieunoh@postech.ac.kr
# ----------------------------------------------------------------

# 0. augment 설정 ------------------------------------
base_path = "/home/guest1/ellen_data/UKB_quality_data2_combined/"  # 여기로 옮길 것
data2move_path = "/home/guest1/ellen_data/UKB_quality_data/"  # 여기 있는 파일을 옮길 것임
num_matchh = 0  # 겹치는 파일 몇개인지 세려고
num_matchl = 0  # 겹치는 파일 몇개인지 세려고

# 1. 겹치는 파일 있는지 확인 --------------------------------
for oldfile in os.listdir(data2move_path+"high_Q"):
    for newfile in os.listdir(base_path+"high_Q"):
        if oldfile == newfile:
            num_matchh += 1
            # print("["+str(num_matchh)+"] HQ File already exist! - "+newfile)

for oldfile in os.listdir(data2move_path+"low_Q"):
    for newfile in os.listdir(base_path+"low_Q"):
        if oldfile == newfile:
            num_matchl += 1
            # print("["+str(num_matchl)+"] LQ File already exist! - "+newfile)

print("=========================================================")
print("there is already ["+str(num_matchh) + "] of hq, [" +
      str(num_matchl)+"] of lq files in the base_path")
print("겹치는 파일 빼고 옮기기 시작")


# 2. high q 파일 복사해서 옮기기 --------------------------------
num_matchh = 0
for oldfile in os.listdir(data2move_path+"high_Q"):
    match = 0  # loop을 새롭게 돌기 시작할 때 다시 false로 바꿔줌
    for newfile in os.listdir(base_path+"high_Q"):
        if oldfile == newfile:
            match = 1  # match하는 것이 있으면match true
            num_matchh += 1
    if not match:  # match가 된게 없으면 파일 옮기기
        shutil.copy(data2move_path+"high_Q/"+oldfile,
                    base_path+"high_Q/"+oldfile)
print("high quality data combined without ["+str(num_matchh)+ "] existing file!")

# 3. low q 파일 복사해서 옮기기 --------------------------------
num_matchl = 0
for oldfile in os.listdir(data2move_path+"low_Q"):
    match = 0
    for newfile in os.listdir(base_path+"low_Q"):
        if oldfile == newfile:
            match = 1
            num_matchl += 1

    if not match:
        shutil.copy(data2move_path+"low_Q/"+oldfile,
                    base_path+"low_Q/"+oldfile)
print("low quality data combined without ["+str(num_matchl)+ "] existing file!")
