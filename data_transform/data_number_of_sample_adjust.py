# import nntplib
# import numbers
# import os
# import shutil
# import numpy as np
# from matplotlib import pyplot as plt
# from PIL import Image

# #___________
# # READ ME-----------------------------------------------------
# # 이거 쓰지말기 - image가 0000-> 0001 순으로 읽히는 것이 아니라서 남은 데이터 명이 깔끔하지 않음 
# # size까지 맞춰져있는 (resize.py) image를 개수 자르기 
# # size 맞춰져있는 폴더 복사해서 !!! 이름 바꾸고 거기서 개수에 맞게 자르기
# # model이 괜찮은지 빨리 확인 할 수 있게
# # 
# # 참고: https://www.delftstack.com/ko/howto/python/python-move-file/
# # 2022.06.22 jieunoh@postech.ac.kr
# # ----------------------------------------------------------------

# # 0. augment 설정 ------------------------------------
# number_of_samples=1000

# # size 맞춰져있는 폴더 복사해서 !!! 이름 바꾸고 거기서 개수에 맞게 자르기
# path = "/home/guest1/ellen_data/UKB_quality_data2_combined/input_20220221_512_n1000"  # 여기로 옮길 것


# # 1. 전체 개수 수에 맞춰서, 비율& 개수 정하기
# trainR, valR, testR= 6,2,2 # train : val: test 비율 쓰기
# htrainR, hvalR, htestRa, htestRb= 6,2,1,1 # train : val: test 비율 쓰기


# htotal_dataset = number_of_samples
# htrainN = int(float(htotal_dataset)/10.*float(htrainR))
# hvalN = int(float(htotal_dataset)/10.*float(hvalR)) 
# htestNa = int(float(htotal_dataset)/10.*float(htestRa)) 
# htestNb = htotal_dataset-htrainN-hvalN-htestNa

# # low qualtiy
# ltotal_dataset = number_of_samples
# ltrainN = int(float(ltotal_dataset)/10.*float(trainR))
# lvalN = int(float(ltotal_dataset)/10.*float(valR)) 
# ltestN = ltotal_dataset -ltrainN-lvalN

# print("ratio of lowquality dataset (train : val : test) : (", trainR, ": ", valR , ": ", testR ,")")
# print("ratio of highquality (train : val : testA: testB) : (", htrainR, ": ", hvalR , ": ", htestRa ,": ",htestRb,")")

# print("High qualtiy # of dataset in (train : val : testA: testB) : (", htrainN, ", ", hvalN , ", ", htestNa ,": ",htestNb,")")
# print("Low qualtiy # of dataset in (train : val : test) : (", ltrainN, ", ", lvalN , ", ", ltestN ,")")

# number = [htrainN, hvalN, htestNb, ltrainN, lvalN]
# folder_name =["trainB/","valB/", "testB/","trainA/","valA/"]

# # 2. 폴더 열어서 위에 말한 개수 이상은 없애기
# # 2-1. testA빼고 다
# for i, folder in enumerate(folder_name):
#     j=0
#     for file in os.listdir(path+"/"+folder):
#         j+=1
#         if j>number[i]:
#             os.remove(path+"/"+folder+"/"+file)
#             print("hi")
#     print(folder+" END>>>>>>>>>>>")        

# # 2-2. testA
# flag=False
# stage =0
# for file in os.listdir(path+"/testA"):
#     if stage==0 and (str(ltestN).zfill(4) in file):
#         stage=1
#     if stage==1:
#         if "h" in file:
#             stage=2
#         else:
#             os.remove(path+"/testA/"+file) # lowquality image자르기
        
#     if stage ==2:
#         if "h"+str(htestNa).zfill(4) in file:
#             stage=3
#     if stage ==3:
#         os.remove(path+"/testA/"+file)
# print("testA END >>>>>>>>>>>>")







