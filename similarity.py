# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################

"""
Created on Thurs Sept 06 10:08:42 2019

@author: Tianpeng

代码功能描述：（1）读取Sharp_waves文件，
            （2）采用巴特沃斯滤波器，进行60-240Hz滤波
            （3）利用欧式距离计算最相似和相异的三对信号，其中没对信号选取300个点
            （4）输出开始的点，以及进行可视化

"""
#####################################################################

import numpy as np
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
start = 113 #从start开始做N个文件的图                    #设立变量start，作为循环读取文件的起始
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


#通过欧式距离得到最相似的三对信号和最相异的信号
##################################################################################################################3
count = 300 #一个信号的个数
max_value = 10000000#这里设定的最大值
sigal_count = int(len(s1_filter1)/count)#一共的信号个数
dis = np.full((sigal_count,sigal_count),0.0)#用于保存欧式距离矩阵
sim,diversity = {},{}#使用字典列表sim，diversity分别保存最相似和相异的三对信号的标号
dis_sim = np.copy(dis)
dis_div = np.copy(dis)

#用于计算两个信号的欧式距离函数
def euclidean_distance(sigs1,sigs2):
    s = 0.0
    for i in range(count):
        s += math.sqrt((sigs1[i] - sigs2[i])**2)
    return s

#使用切片传递信号，调用欧式距离函数，并且得到欧式距离矩阵
for row in range(sigal_count):
    i = row+1
    while i < sigal_count:
        temp = euclidean_distance(s1_filter1[row*count:(row+1)*count],s1_filter1[i*count:(i+1)*count])
        #同时更新对应的距离矩阵
        dis[row,i] = dis[i,row] = temp
        i += 1

#找最相异的三对,并且保存在字典中
def fuc_div():
    for i in range(3):
        pos = np.unravel_index(np.argmax(dis_div), dis_div.shape)
        dis_div[pos[0],pos[1]] = dis_div[pos[1],pos[0]] = 0.0
        index = "div" + str(i)
        diversity[index] = pos

#找最相似的三对,并且保存在字典中
def fuc_sim():
    # 将所有的对角元素设置为最大值
    for i in range(sigal_count):
        dis_sim[i,i] = max_value
    for i in range(3):
        pos = np.unravel_index(np.argmin(dis_sim), dis_sim.shape)
        dis_sim[pos[0],pos[1]] = dis_sim[pos[1],pos[0]] = max_value
        index = "sim" + str(i)
        sim[index] = pos

#输出最相似和最相异的三对信号的起始点
def output_point():
    print(sim)
    print(diversity)
    print("最相似的三个信号的起始点")
    i=0
    for name, pointers in sim.items():
        i=0
        for pointer in pointers:
            if i==0:
                print(name)
                print("第一个点："+str(count*pointers[0]) + "\n" +"第二个点："+str(count*pointers[1]) )
                i=1
    print("最相异的三个信号的起始点")
    for name1, pointers1 in diversity.items():
        i=0
        for pointer1 in pointers1:
            if i==0:
                print(name1)
                print("第一个点："+str(count*pointers1[0]) + "\n" +"第二个点："+str(count*pointers1[1]) )
                i=1

###################################################################################################################


#画图,画四行三列
###################################################################################################################
def fuc_draw():
    fig1 = plt.figure()         #创建画图对象，开始画图
    plt.subplots_adjust(hspace=0.35)
    positon = [1,2,3]
    i=0
    for name,pointers in sim.items():#142536
        for pointer in pointers:
            ax = fig1.add_subplot(2, 3, positon[i])  #让相似或相异的图想为上下排列
            plt.plot(s1_filter1[pointer*count:pointer*count+count],color="red")
            ax.set_title(name)
            positon[i] += 3
        i += 1
    fig2 = plt.figure()         #创建画图对象，开始画图
    plt.subplots_adjust(hspace=0.35)
    j=0
    positon1 = [1, 2, 3]
    for name,pointers in diversity.items():
        for pointer in pointers:
            ax1 = fig2.add_subplot(2, 3, positon1[j])  #让相似或相异的图想为上下排列
            plt.plot(s1_filter1[pointer*count:pointer*count+count],color="blue")
            ax1.set_title(name)
            positon1[j] += 3
        j += 1

    plt.show()
    plt.close()        #
    #关闭绘图对象，释放绘图资源

#调用函数即可
dis_div = dis.copy()
dis_sim = dis.copy()
fuc_div()
fuc_sim()
output_point()
fuc_draw()
##################################################################################################################


