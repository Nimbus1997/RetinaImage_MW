#!/usr/bin/env python
# coding: utf-8

# ## Illumination - AQE 
# by H.Bartling ‘Automated Quality Evaluation of Digital Fundus Photography’ 2009 Acta Ophthalmologica (impact factor: 3.761, # cite: 68)
# 
# ------------------------------------------------
# read me
# > 핵심 def : illumination(img:np.array, num: int)->float
# > 2022.06.23 jieunoh@postech.ac.kr
# ------------------------------------------------
# # 0. Environment setting
# 1. Image read
# 2. Normalize(size is alread normalized) & gray scale
# 3. Split the fundus images 
# ---
# 4. Calculate Brightness : B
# 5. Calculate Contrast: C
# 6. I = B+C , I>1.3 : not acceptable, I<=1.3 : acceptable
# 7. Illumination value: fraction of all squares in an image classified as acceptable

# In[23]:


import numpy as np
import os
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt


# In[24]:


def min_110(a):
    #return minimum value between the value and 110
    if 0.5<a<110:
        return a
    elif a<=0.5:
        return 0.5
    else:
        return 110


# In[25]:


def min_8(a):
    # return the input value if the input value is same or smaller than 8, and 30 if bigger
    if a>8:
        return 30
    elif a>0.5:
        return a
    else:
        return 0.5


# In[36]:


def brightness(sub):
    m=np.mean(sub)
    # print("mean:", m)
    return ((110-m)/min_110(m))**2


# In[37]:


def contrast(sub):
    s=np.std(sub)
    # print("std:", s)
    return ((8-s)/min_8(s))**2


# In[38]:


def illumination_ac(sub_i):
    # input: float
    # output: bool. if i>1.3: false(not acceptable), other wise: true(acceptable)
    if sub_i>1.3:
        return False
    else: 
        return True


# In[41]:


def illumination(img:np.array, num: int)->float:
    # illumination value for each image -> need to see each subimg
    # num = size_ori/size_sub
    # input : image(np.array), sub image number(int)(한 모서리기준)
    # output : each image의 최종 illumination value(float)
    count = 0
    for i, v in enumerate(np.vsplit(img,num)):
       for j, sub in enumerate(np.hsplit(v, num)):
        # locals() ["subimg_{}{}".format(i,j)] = sub # https://trustyou.tistory.com/m/197
        
        sub_b=brightness(sub)
        sub_c=contrast(sub)
        sub_i=sub_b+sub_c
        # print(i,j)
        # print("sub.shape: ",sub.shape)
        # print("sub_b:",sub_b)
        # print("sub_c:",sub_c)
        
        # print("sub_i:",sub_i)
        if illumination_ac(sub_i):
            count+=1
    # print("_______")
    # print(count)
    return count/(num**2)
