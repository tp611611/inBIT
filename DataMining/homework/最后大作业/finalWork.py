# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################

"""
Created on Fri Nov 29 10:08:42 2019

@author: Tianpeng

代码功能描述：（1）读取类别数据，得到类别图
            （2）读入脑电数据，进行滤波处理，计算相应的特征数据，把得到的特征数据作为feature_all.xlsx文件存储下来。
            （3）读取存储的xlsx文件和标签数据（其中删除了只有一个标签的数据，认为为不重要异常数据），进行随机数据划分，划分为测试集和训练集
            （4）利用训练数据，训练随机森林模型，参数优化使用网络搜索优化，得到认为最好的随机森林参数
            （5）将得到的最好参数模型对测试数据集进行评价，输出混淆矩阵等。
"""
#####################################################################

import matplotlib.pyplot as plt                      # 进行绘图的相关模块
from scipy import signal                             # SciPy的signal信号处理用子模块
import pandas as pd                                  # 数据框数据处理分析
import numpy as np                                   # 数组数据处理分析
import math                                          # 数学运算模块
from sklearn.model_selection import train_test_split # 进行数据随机划分模块
from sklearn.model_selection import GridSearchCV    # 网络搜索模块用于优化参数
from sklearn import metrics                         # 用于得到评价信息
from sklearn.ensemble import RandomForestClassifier # 用于引入随机森林模型
import warnings                                     # python警告模块
warnings.filterwarnings("ignore")



def toGetLabelData():                                       # 得到类别数据
    classData = open(r'./data/sc4002e0_label.txt').read()   # classData为类别数据定义
    classData = classData.split()
    classData = [int(float(i)) for i in classData]
    del classData[-1]                                       # 去掉标签末尾无对应数据
    return classData

def drawLabel(tempData):                                     # 根据类别数据画出饼图
    lebelCount =[0,0,0,0,0,0,0]
    for i in tempData:
        lebelCount[i] += 1
    lables = ['W','S1','S2','S3','S4','R','M']
    plt.pie(lebelCount,labels=lables)
    plt.show()
    plt.close()

def del_signal(orignal_data):                               # 滤波函数
    fs = 3000                                               # 设立频率变量fs
    lowcut = 1
    highcut = 30
    order = 2                                               # 设立滤波器阶次变量
    nyq = 0.5 * fs                                          # 设立采样频率变量nyq，采样频率=fs/2。
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')  # 设计巴特沃斯带通滤波器 “band”
    after_data = signal.lfilter(b, a, orignal_data)         # 将s1带入滤波器，滤波结果保存在s1_filter1中
    return after_data                                       # 返回滤波后的数据

def toGetData():                                             # 得到脑电数据,进行滤波，并返回滤波后的数据，画图进行比较
    orignal_data = open(r'./data/sc4002e0_data.txt').read()  # 读取了脑电波信号
    orignal_data =  orignal_data.split()
    orignal_data = [float(i) for i in orignal_data]
    data = del_signal(orignal_data)
    fig1 = plt.figure()                                      # 创建画图对象，开始画图
    ax1 = fig1.add_subplot(211)
    plt.plot(orignal_data,color='r')                         # 在选定的画布位置上，画未经滤波的s1图像，设定颜色为红色
    ax1.set_title('Denoised Signal')                         # 设定子图211的title为denoised signal
    plt.ylabel('Amplitude')                                  # 设定子图211的Y轴lable为amplitude
    ax2 = fig1.add_subplot(212)
    plt.plot(data,color='blue')                              # 在选定的画布位置上，画经过滤波的s1_filter1图像，设定颜色为红色
    ax2.set_title('Denoised Signal')                         # 设定子图212的title为denoised signal
    plt.ylabel('Amplitude')                                  # 设定子图212的Y轴lable为amplitude
    plt.show()
    plt.close('all')                                         # 关闭绘图对象，释放绘图资源
    return data

def to_get_feature(data):                                    # 提取特征，保存为xlsx表，然后再返回该数据框
    data_columns = ['means1', 'max_value2', 'variance3', 'zero_count4', 'skewness5'
        , 'kurtosis6', 'petrosian7','middle8','min_value9']
    data_excel = pd.DataFrame(np.zeros((2830, 9)), columns=data_columns)
    frequncy_count = int(len(data) / 2830)
    for i in range(2830):
        temp = data[i * frequncy_count:(i + 1) * frequncy_count]
        means_temp = sum(temp) / frequncy_count              # 1计算均值
        max_temp = max(temp)                                 # 2计算最大值
        variance_temp = 0.0                                  # 3计算方差
        zero_count_temp = 0.0                                # 4过零点次数
        skewness_temp = 0.0                                  # 5计算偏度
        kurtosis_temp = 0.0                                  # 6计算峰度
        petrosian_temp = 0.0                                 # 7计算分形维度
        middle_temp = np.median(temp)                        # 计算中值
        min_temp = min(temp)                                 # 计算最小值
        petrosian_tag1 = 0.0                                 # 保存0的次数，用经过零点的次数减去该数就是信号符号变化的次数
        for j in range(frequncy_count):
            if temp[j] == 0:
                zero_count_temp += 1
                petrosian_tag1 += 1
            elif j > 0:
                if temp[j - 1] * temp[j] < 0:
                    zero_count_temp += 1
            variance_temp += (temp[j] - means_temp) ** 2 / frequncy_count
        s = pd.Series(temp)
        skewness_temp = s.skew()
        kurtosis_temp = s.kurt()
        petrosian_temp = math.log(frequncy_count, 10) / (math.log(frequncy_count, 10) +
                                                         math.log(frequncy_count / (frequncy_count + 0.4 * (
                                                                     zero_count_temp - petrosian_tag1)), 10))
        data_excel.loc[i] = [means_temp, max_temp, variance_temp, zero_count_temp, skewness_temp, kurtosis_temp,
                             petrosian_temp,middle_temp,min_temp]
    data_excel.to_excel('./data/feature_all.xlsx', sheet_name='Sheet1')



def to_divide_data():                                    # 读取对应的xlxs文件，然后划分和返回测试数据集和训练数据集
    target_data_numpy = toGetLabelData();
    train_data_frame = pd.read_excel("./data/feature_all.xlsx")
    index = 0                                           # 记录只有一个数据M期标签为6的索引
    for i in range(len(target_data_numpy)):             # 删除只有一个数据的M数据
        if target_data_numpy[i] == 6 :
            index = i
            break
    del target_data_numpy[index]
    train_data_frame = train_data_frame.drop([index])
    colnums_train_data = list(train_data_frame)
    train_data_frame = train_data_frame.drop([colnums_train_data[0]],axis=1)
    feature_train, feature_test, target_train, target_test = train_test_split(train_data_frame, target_data_numpy,
                                                                       test_size=0.2, random_state=0)
    return feature_train,feature_test,target_train,target_test

def to_train_model(train_data_x,train_data_y):           # 利用网络搜索优化参数，得到最优的训练模型，并且返回最优的训练模型
    n_estimators = [100,135,170,200,225]
    max_depth = list(range(6,9))
    rfgs = RandomForestClassifier(oob_score=True,random_state=1)
    para_grid = [{"n_estimators":n_estimators,"max_depth":max_depth}]
    gs_rf = GridSearchCV(estimator=rfgs,param_grid=para_grid,n_jobs=3)
    all_gs_rfs = gs_rf.fit(train_data_x,train_data_y)
    best_rf_params = all_gs_rfs.best_params_
    print(all_gs_rfs.best_score_)
    print(best_rf_params)
    return best_rf_params

def to_evalate_model(rf_best_params,test_data_x,test_data_y,train_x,train_y):      # 评价模型得到混淆矩阵输出相关信息
    final_rf = RandomForestClassifier(n_estimators=rf_best_params['n_estimators'],max_depth=rf_best_params['max_depth'])
    final_rf.fit(train_x,train_y)
    y_model_predict = final_rf.predict(test_data_x)
    print("预测的正确率为：",round(metrics.accuracy_score(test_data_y,y_model_predict),4)*100,"%")
    confusion_mat = metrics.confusion_matrix(test_data_y,y_model_predict)           # 输出混淆矩阵
    class_label = ['class-0','class-1','class-2','class-3','class-4','class-5']
    confusion_mat_dataframe = pd.DataFrame(confusion_mat,columns=class_label,index=class_label)
    print("             混淆矩阵输出为：" )
    print(confusion_mat_dataframe)
    print(metrics.classification_report(test_data_y,y_model_predict,target_names = class_label))# classification_report

# classData = toGetLabelData()                                                      # 得到标签数据
# drawLabel(classData)                                                              # 绘制类别图
train_x,test_x,train_y,test_y = to_divide_data()                                    # 进行数据划分，train_x:训练数据集 train_y训练数据标签 test_x:测试数据集，test_y：测试数据集标签
rf_best_params = to_train_model(train_x,train_y)                                    # 进行模型训练，同时使用网络搜索找到更合适的参数并且返回，rf_best_params：最优模型的参数
to_evalate_model(rf_best_params,test_x,test_y,train_x,train_y)                      # 得到模型评价，输出混淆矩阵

