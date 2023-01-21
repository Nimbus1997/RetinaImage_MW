import os
import shutil

#폴더 명 바꿔주기
source_dir_path='./high_Q'
new_dir='cycleGAN_highQ_only_1'


#폴더 생성
dirlist=['trainA','trainB','valA','valB','testA', 'testB']
for d in dirlist:
    print(d)
    if not os.path.exists("./"+new_dir+"/"+d):
        os.makedirs("./"+new_dir+"/"+d)

# --------------
# 파일 복사 및 이름 변경- 1) A, B same file
# 1. 파일 읽기
i =0 
# for file in os.listdir(source_dir_path):
#     source = source_dir_path+"/"+file
#     if i <1500:
#         trainA = "./"+new_dir+"/trainA/"+"_".join([file.split('.')[0],"A.png"])
#         shutil.copy(source,trainA)

#         trainB = "./"+new_dir+"/trainB/"+"_".join([file.split('.')[0],"B.png"])
#         shutil.copy(source,trainB)
#     elif i <2100:
#         valA = "./"+new_dir+"/valA/"+"_".join([file.split('.')[0],"A.png"])
#         shutil.copy(source,valA)

#         valB = "./"+new_dir+"/valB/"+"_".join([file.split('.')[0],"B.png"])
#         shutil.copy(source,valB)
#     else:
#         testA = "./"+new_dir+"/testA/"+"_".join([file.split('.')[0],"A.png"])
#         shutil.copy(source,testA)
#         testB = "./"+new_dir+"/testB/"+"_".join([file.split('.')[0],"B.png"])
#         shutil.copy(source,testB)
#     i+=1

for file in os.listdir(source_dir_path):
    source = source_dir_path+"/"+file
    if i <1500:
        trainA = "./"+new_dir+"/trainA/"+"_".join([str(i),"A.png"])
        shutil.copy(source,trainA)

        trainB = "./"+new_dir+"/trainB/"+"_".join([str(i),"B.png"])
        shutil.copy(source,trainB)
    elif i <2100:
        valA = "./"+new_dir+"/valA/"+"_".join([str(i),"A.png"])
        shutil.copy(source,valA)

        valB = "./"+new_dir+"/valB/"+"_".join([str(i),"B.png"])
        shutil.copy(source,valB)
    else:
        testA = "./"+new_dir+"/testA/"+"_".join([str(i),"A.png"])
        shutil.copy(source,testA)
        testB = "./"+new_dir+"/testB/"+"_".join([str(i),"B.png"])
        shutil.copy(source,testB)
    i+=1





