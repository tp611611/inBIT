import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#数据预处理 数据实现中心化Z-Score
def create_data():                                        # 先读入数据，从上次作业中得到的xlsx(已打标签上得到)
    dataSet_dataframe = pd.read_excel('./homework2.xlsx')
    standardScaler = StandardScaler()
    standardScaler.fit(dataSet_dataframe)
    dataSet_dataframe_standard = standardScaler.transform(dataSet_dataframe)
    dataSet_dataframe_standard = dataSet_dataframe_standard[:,1:8]
    return dataSet_dataframe_standard

#根据数据的个数随机产生随机数列表
def random_k(data,k):
    set_temp = set()  #用来判断是否产生相同的随机数
    list_random = []
    while len(list_random)<k:
        temp = random.randint(0, len(data)-1)
        if temp not in set_temp:
            set_random = set()
            set_random.add(temp)
            list_random.append(list(data[temp]))
        set_temp.add(temp)
    return list_random

#使用欧式距离
def cal_euclidean_distance(datavector1,datavector2):         #计算每两个样本的距离,返回一个数
    dis_value = 0.0
    for i in range(len(datavector1)):
        dis_value += (datavector1[i] - datavector2[i])**2
    return  dis_value

#将某个元素划分为一个集合
def simple_div(data,i,list_means,k):
    j = 0
    class_collection = 0
    min_dis = float("inf")
    while j < k:
        dis_temp = cal_euclidean_distance(data[i,:],list_means[j]);
        if(dis_temp < min_dis):
            class_collection = j
            min_dis = dis_temp
        j += 1
    return  class_collection

#产生K个空集合
def create_set(k):
    myset = list()
    for i in range(k):
        myset.append(set())
    return myset

#更新均值点
def update_list_means(myset, data_train, k,list_means):
    for i in range(k):#第几个集合
        for k in range(data_train.shape[1]):  # 每个元素的维数
            dim_sum = 0.0
            for j in myset[i]:#每个集合中元素
                dim_sum += data_train[j][k]
            list_means[i][k] = dim_sum/len(myset[i])
    return list_means

def cal_e(list_means, data_train, myset,k):
    sum_e =0.0
    #计算每个集合中 元素与该对应中心点欧式距离 然后计算总和
    for i in range(k):
        #每个集合中元素
        for j in myset[i]:
            sum_e += cal_euclidean_distance(data_train[j,:],list_means[i])
    return  sum_e


def set_to_null(myset,k):
    for i in range(k):
        myset[i] = set()
    return myset

def main():
    data_train = create_data()
    k = 3
    sun_e = float('inf')
    sun_old = 0.0
    # 产生k个随机数据点
    list_means = random_k(data_train,k)
    #产生k个空集合，使用列表集合嵌套，保存下标
    myset = create_set(k)
    myset_ture = None  #保存集合分布情况
    count = 0#保存迭代次数
    while sun_e!=0:
        #计算每个数据的分类
        for i in range(len(data_train)):
            simple_class = simple_div(data_train,i,list_means,k)
            myset[simple_class].add(i)
        myset_ture = myset.copy()
        #更新随机点坐标，即均值点
        list_means = update_list_means(myset,data_train,k,list_means)
        #计算平均误差和
        sun_new = cal_e(list_means,data_train,myset,k)
        sun_e = sun_new-sun_old
        sun_old = sun_new
        # 所有的集合重新置为空集
        myset = set_to_null(myset,k)
        count += 1
    print("k为：    "+str(k))
    print("次数为"+ str(count))
    print("集合分布为：   ")
    print(myset_ture )
main()
