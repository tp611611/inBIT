# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################

"""
Created on Fri Jan 06 10:08:42 2017

@author: Yuyangyou

代码功能描述：（1）读取Sharp_waves文件，
              （2）采用巴特沃斯滤波器，进行60-240Hz滤波
              （3）画图
              （4）....

"""
#####################################################################

import numpy_myself as np
from scipy import signal
import math
import matplotlib
import matplotlib.pylab as plt #绘图
#from numpy import *
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

#画图
###################################################################################################################
    fig1 = plt.figure()                             #创建画图对象，开始画图
    ax1 = fig1.add_subplot(211)
                    #在一张figure里面生成多张子图，将画布分割成2行1列， 图像画在从左到右从上到下的第1块
                    #例如，fig1.add_subplot(349)  将画布分割成3行4列，图像画在从左到右从上到下的第9块

    plt.plot(s1,color='r')                          #在选定的画布位置上，画未经滤波的s1图像，设定颜色为红色
    ax1.set_title('Denoised Signal')               #设定子图211的title为denoised signal
    plt.ylabel('Amplitude')                         #设定子图211的Y轴lable为amplitude

    ax2 = fig1.add_subplot(212)
                    # 在一张figure里面生成多张子图，将画布分割成2行1列， 图像画在从左到右从上到下的第2块

    plt.plot(s1_filter1,color='blue')                  #在选定的画布位置上，画经过滤波的s1_filter1图像，设定颜色为红色
    ax2.set_title('Denoised Signal')               #设定子图212的title为denoised signal
    plt.ylabel('Amplitude')                         #设定子图212的Y轴lable为amplitude
    # plt.savefig(r'./data/20151026_%d.png' % (e))  #保存图像，设定保存路径并统一命名，%d处，十进制替换为e值

    plt.show()

    plt.close('all')                                 #关闭绘图对象，释放绘图资源
##################################################################################################################







































































