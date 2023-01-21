import os
import shutil

#폴더 명 바꿔주기
source_dir_path='./high_Q'
new_dir='pix2pix_highQ_only'


#폴더 생성
dirlist=['train','val','test']
for d in dirlist:
    print(d)
    if not os.path.exists("./"+new_dir+"/"+d):
        os.makedirs("./"+new_dir+"/"+d)

# --------------
# 파일 복사 및 이름 변경- 1) A, B same file
# 1. 파일 읽기
i =0 
for file in os.listdir(source_dir_path):
    source = source_dir_path+"/"+file
    if i <1500:
        train = "./"+new_dir+"/train/"+file
        shutil.copy(source,train)
    elif i <2100:
        val = "./"+new_dir+"/val/"+file
        shutil.copy(source,val)
    else:
        test = "./"+new_dir+"/test/"+file
        shutil.copy(source,test)
    i+=1




