# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################

"""
Created on Fri Oct 23 10:08:42 2019

@author: Tianpeng

代码功能描述：（1）读取第二次xlsx文件,该文件已经人工加上了属性类标签，划分数据集为训练集和测试集
            （2）调用相关函数得到测试数据的分类结果
            （3）用测试集得到混淆矩阵并且输出正确率
            （4）调用相关函数得到结果

"""
#####################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_data():                                        # 先读入数据，从上次作业中得到的xlsx(已打标签上得到)
    dataSet_dataframe = pd.read_excel('./homework2.xlsx')
    dataSet = []
    for row in dataSet_dataframe.itertuples():
        row1 = list(row)
        del row1[0:2]
        dataSet.append(row1)
    return dataSet
def cal_euclidean_distance(datavector1,datavector2):         #计算每两个向量的距离,返回一个数
    dis_value = 0.0
    for i in range(len(datavector1)):
        dis_value += (datavector1[i] - datavector2[i])*2
    return  dis_value

def cal_distance(train_data,data_vector):                    #计算每一个向量和训练集的距离函数，返回该距离列表
    dis_list = []
    for vector in train_data:
        dis_list.append(cal_euclidean_distance(vector,data_vector))
    return  dis_list

def vote_majority(train_data,index_list):                   #多数投票原则函数
    dict_class = {}
    for index in index_list:
        if train_data[index][-1] not in dict_class:
            dict_class[train_data[index][-1]] = 0
        dict_class[train_data[index][-1]] += 1
    return max(dict_class,key=dict_class.get)

def get_class(train_data,vector,k):                         #返回对该单个测试数据的分类
    vector_distance = cal_distance(train_data,vector)
    index_list = []                                         #找到前7个距离最小的索引值
    while len(index_list) < k:
        min_index = vector_distance.index(min(vector_distance))
        index_list.append(min_index)
        vector_distance[min_index] = float('inf')
    vector_class = vote_majority(train_data,index_list)
    return  vector_class

def knn_main():
    # k_list = list((2,3,4,5,6,7,8,9,10,11,12,13,14,15))
    #     # for k in k_list:
    k = 9
    confusion_matrix = np.zeros((3, 3))
    mydataset = create_data()
    train_data,test_data = train_test_split(mydataset,test_size=0.3)
    for vector in test_data:
        vector_class = get_class(train_data,vector,k)
        confusion_matrix[vector[-1]][vector_class] += 1
    confusion_matrix_dataframe = pd.DataFrame(confusion_matrix, index=
    ['本来的类型为0:  ', '本来的类型为1:  ', '本来的类型为2:  '], columns=['预测类型0:  ', '预测类型1: ', '预测类型2:  '])
    print("当k等于  " + str(k) + "  时的混淆矩阵为：")
    print(confusion_matrix_dataframe)
    sum = 0.0  # 保存正确率
    for i in range(3):
        sum += confusion_matrix[i][i]
    print("预测的正确率为：" + str(sum / len(test_data) * 100) + '%')  # 输出正确率
knn_main()