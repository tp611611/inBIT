# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################

"""
Created on Fri Oct 06 10:08:42 2019

@author: Tianpeng

代码功能描述：（1）读取Sharp_waves文件，
              （2）采用巴特沃斯滤波器，进行60-240Hz滤波
              （3）采用数据框结构，计算相应的属性包括：1.均值 2.最大值 3.方差 4.过零点次数 5.偏度 6.峰度 7. Petrosian分形维数
                8. Renyi熵（即α=1时香农熵） 9.近似熵
                最后得到相应的数据框表格
              （4）保存为excel格式并且进行输出

"""
#####################################################################

import numpy as np
import pandas as pd
from scipy import signal
import math
import matplotlib
import matplotlib.pylab as plt #绘图
import os
import gc   #gc模块提供一个接口给开发者设置垃圾回收的选项
import time

#读取文件第一列，保存在s1列表中
###########################################################################################################
start = 113 #从start开始做N个·文件的图                    #设立变量start，作为循环读取文件的起始
N = 1                                                      #设立变量N，作为循环读取文件的增量
for e in range(start,start+N):                            #循环2次，读取113&114两个文件
    data = open(r'./data/20151026_%d'% (e)).read()     #设立data列表变量，python 文件流，%d处，十进制替换为e值，.read读文件
    data = data.split( )                                  #以空格为分隔符，返回数值列表data
    data = [float(s) for s in data]                       #将列表data中的数值强制转换为float类型

    s1 = data[0:45000*4:4]                          #list切片L[n1:n2:n3]  n1代表开始元素下标；n2代表结束元素下标
                                                    #n3代表切片步长，可以不提供，默认值是1，步长值不能为0
####################################################################################################################


#滤波
##################################################################################################################
    fs = 3000                                           #设立频率变量fs
    lowcut = 1
    highcut = 30
    order = 2                                           #设立滤波器阶次变量
    nyq = 0.5*fs                                        #设立采样频率变量nyq，采样频率=fs/2。
    low = lowcut/nyq
    high = highcut/nyq
    b,a = signal.butter(order,[low,high],btype='band') #设计巴特沃斯带通滤波器 “band”
    s1_filter1 = signal.lfilter(b,a,s1)                 #将s1带入滤波器，滤波结果保存在s1_filter1中
###################################################################################################################


#3计算该表（30行7列）
###################################################################################################################
data_columns=['means1','max_value2','variance3','zero_count4','skewness5'
    ,'kurtosis6','petrosian7']
data_excel = pd.DataFrame(np.zeros((30,7)),columns=data_columns)
frequncy_count = int(len(s1_filter1)/30)
for i in range(30):
        temp = s1_filter1[i*frequncy_count:(i+1)*frequncy_count]
        means_temp = sum(temp)/frequncy_count   #1计算均值
        max_temp = max(temp)    #2计算最大值
        variance_temp = 0.0     #3计算方差
        zero_count_temp = 0.0   #4过零点次数
        skewness_temp = 0.0     #5计算偏度
        kurtosis_temp = 0.0     #6计算峰度
        petrosian_temp = 0.0    #7计算分形维度
        petrosian_tag1 = 0.0   #保存0的次数，用经过零点的次数减去该数就是信号符号变化的次数
        for j in range(frequncy_count):
            if temp[j]==0:
                zero_count_temp += 1
                petrosian_tag1 += 1
            elif j>0:
                if temp[j-1]*temp[j] < 0:
                    zero_count_temp += 1
            variance_temp += (temp[j] - means_temp)**2/frequncy_count
        s = pd.Series(temp)
        skewness_temp = s.skew()
        kurtosis_temp = s.kurt()
        petrosian_temp = math.log(frequncy_count,10)/(math.log(frequncy_count,10)+
        math.log(frequncy_count/(frequncy_count+0.4*(zero_count_temp-petrosian_tag1)),10))
        data_excel.loc[i]= [means_temp,max_temp,variance_temp,zero_count_temp,skewness_temp,kurtosis_temp,petrosian_temp]
###################################################################################################################


#4输出该excel表并且保存（30行7列）
###################################################################################################################
print(data_excel)
data_excel.to_excel('./homework2.xlsx', sheet_name='Sheet1')