import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
matplotlib.rcParams['font.family']='SimHei'                             # 用来正常显示中文
plt.rcParams['axes.unicode_minus']=False                                # 用来正常显示负号
decisionNode = dict(boxstyle="sawtooth", fc="0.8")                      #绘图相关参数设置
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def create_data():                                                       # 先读入数据，从上次作业中得到的xlsx(已打标签上得到)
    dataSet_dataframe = pd.read_excel('./homework2.xlsx')
    dataSet = []
    labels = ['means1', 'max_value2', 'variance3', 'zero_count4',
              'skewness5' , 'kurtosis6', 'petrosian7']
    # 把数据框变为嵌套链表
    for row in dataSet_dataframe.itertuples():
        row1 = list(row)
        del row1[0:2]
        dataSet.append(row1)
    return dataSet, labels                                              #返回一个数组和一个类标签

def split_datatset_cart(data_vector, index_feature, value):             # 划分数据集，根据传入特征索引值和value，返回小于和大于两个数据集
    split_set_less = []                                                 # 判别为“小于”的子数据集
    split_set_more = []                                                 # 判别为“大于”的子数据集
    for vector in data_vector:
        if vector[index_feature] <= value:                              # 去掉第index_feature列特征
            split_1 = vector[:index_feature]
            split_1.extend(vector[index_feature + 1:])
            split_set_less.append(split_1)
        else:
            split_2 = vector[:index_feature]
            split_2.extend(vector[index_feature + 1:])
            split_set_more.append(split_2)                              #分别返回D1和D2数据集以及对应的数据集样本数
    return len(split_set_less), split_set_less, len(split_set_more), split_set_more

def max_class(label_list):                                              #返回类列表中出现次数最多的类标签
    count_label = {}
    for label in label_list:
        if label not in count_label:
            count_label[label] = 0
        count_label[label] += 1
    return max(count_label, key=count_label.get)                        #选择字典value最大的所对应的key值即类别

def choose_bestfeture_cart(data_vector):                                #为当前数据集寻找最优特征和最优划分点
    nums_data = len(data_vector)
    nums_feature = len(data_vector[0]) - 1                              # 每个样本所包含的特征个数
    min_gini = float('inf')                                             # 表示最小的基尼指数
    best_index_feature = 0                                              # 表示最优特征的索引位置index
    best_split_point = None                                             # 表示最优的切分点
    for i in range(nums_feature):                                       # 遍历所有的特征
        features_i_set = [vector[i] for vector in data_vector]          # 提取第i个特征中所包含的可能取值
        features_i_set.sort()                                           # 对该数据值进行排序
        feature_gini = 0                                                # 每个特征中每个特征值所对应的基尼指数
        for j in range(int(len(features_i_set)) - 1):                   # 得到每个特征下子集集合的gini系数
            nums_less, data_set_less, nums_more, data_set_more = split_datatset_cart(data_vector, i, (
                        features_i_set[j] + features_i_set[j + 1]) / 2)
            gini_temp_less = cal_gini(data_set_less)
            gini_temp_more = cal_gini(data_set_more)
            gini_temp = (nums_less / len(data_vector)) * gini_temp_less + (
                        nums_more / len(data_vector)) * gini_temp_more
            if gini_temp < min_gini:
                best_index_feature = i
                best_split_point = (features_i_set[j] + features_i_set[j + 1]) / 2
                min_gini = gini_temp
                if min_gini <= 0.25:                                    # 当gini系数小于0.15时，直接选择这个点作为分割点
                    return best_index_feature, best_split_point
    return best_index_feature, best_split_point                         # 返回最优分类特征的索引位置和最优切分点

def cal_gini(data_vector):                                              # 计算基尼指数,计算一个数据列表的的gini系数
    nums_data = len(data_vector)                                        # 数据集样本数
    counts_by_labels = {}                                               # 用来保存每个label下的样本数
    gini = 0                                                            # 基尼指数
    p_sum = 0                                                           # 每个类别的样本数

    for vector in data_vector:
        if vector[-1] not in counts_by_labels:                          # vector[-1]为label值
            counts_by_labels[vector[-1]] = 0
        counts_by_labels[vector[-1]] += 1                               # 统计label出现的次数
    for key in counts_by_labels:
        p = float(counts_by_labels[key] / nums_data)                    # 计算每个标签出现的概率
        p_sum += p ** 2
    gini = 1 - p_sum
    return gini

def predect_simple(data_row, cart_tree,labels):                         #利用决策树字典，输入单个元组，得到该元组的分类结果
    class_label = -1                                                    #用class_label来保存该元组的分类结果
    while class_label == -1:
        for key, value in cart_tree.items():                            #先是属性值，value有可能是值，同时也可能是一个嵌套字典
            m = labels.index(key)
            cart_tree = cart_tree[key]
            keys = list(cart_tree.keys())
            key_temp = keys[0]
            list_value = key_temp.split(" ")
            feature_value = float(list_value[-1])
            if feature_value < data_row[m]:
                cart_tree = cart_tree[keys[0]]
            else:
                cart_tree = cart_tree[keys[-1]]
            if type(cart_tree)!=dict:
                class_label = cart_tree
                break
    return class_label                                                   #返回该元组的决策树分类结果

class Decision_tree():                                                   #决策树的生成
    def __init__(self, data_vector, labels):
        self.data_vector = data_vector                                   #数据集
        self.labels = labels                                             # 特征标签
        self.best_feature_index_list = []                                # 用于保存最优特征的索引信息，列表形式输出

    def tree_main(self):                                                 # 生成决策树，返回决策树tree，字典形式
        tree = self.create_decision_tree(self.data_vector, self.labels)
        return tree

    def create_decision_tree(self, data_vector, labels):                 #递归产生用嵌套字典表示的cart决策树
        nums_label = [vector[-1] for vector in data_vector]              #递归停止条件1：如果数据集中所有实例属于同一个类，返回该类标签。
        if len(set(nums_label)) == 1:
            return nums_label[0]
        if len(data_vector[0]) == 1:                                     #递归停止条件2：遍历完特征时，返回出现次数最多的类标签
            return max_class(nums_label)
        best_index_feature, best_split_point = choose_bestfeture_cart(data_vector)      # 选择最优特征
        self.best_feature_index_list.append((best_index_feature, best_split_point))
        best_feature_label = labels[best_index_feature]                                 # 最优特征的标签
        myTree = {best_feature_label: {}}                                # 子决策树，key为最优特征的标签，value为子决策树
        del (labels[best_index_feature])                                 # 删除已经使用过的最优特征标签
        nums_data_left, data_set_left, nums_set_right, data_set_right\
            = split_datatset_cart(data_vector, best_index_feature,  best_split_point)
        myTree[best_feature_label]['<= '+str(best_split_point           # 递归产生左子树
                                             )] = self.create_decision_tree(data_set_left, labels[:])
        myTree[best_feature_label]['> '+str(best_split_point            # 递归产生有子树
                                            )] = self.create_decision_tree(data_set_right, labels[:])
        return myTree                                                   #返回生产的cart决策树(以嵌套字典表示)


def getNumLeafs(myTree):                                                #用于绘制决策树函数，得到该树的叶子总数
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):                                               #得到该树的最大深度
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):                    #centerPt节点中心坐标  parentPt 起点坐标
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):                       #在两个节点之间的线上写上字
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=18)


def plotTree(myTree, parentPt, nodeTxt):                            # 画树
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(myTree):                                             #画出决策树的主函数
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    plotTree.totalW = float(getNumLeafs(myTree))
    plotTree.totalD = float(getTreeDepth(myTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(myTree, (0.5, 1.0), '')
    plt.show()

dataSet, labels = create_data()                                         # 调用函数得到数据集和标签
x_train, x_test = train_test_split(dataSet, test_size=0.3,              # 划分训练集和测试集
                                   random_state=0)
cart_tree = Decision_tree(x_train, labels[:])                           #得到Decision_tree对象
mytree = cart_tree.tree_main()                                          #得到mytree决策树
confusion_matrix = np.zeros((3,3))                                      # 根据mytree字典来测试测试集得到混淆矩阵
for test_vector in x_test:
    class_value = predect_simple(test_vector,mytree.copy(),labels)
    confusion_matrix[int(test_vector[-1])][int(class_value)] += 1
confusion_matrix_dataframe = pd.DataFrame(confusion_matrix,index=
['本来的类型为0:  ','本来的类型为1:  ','本来的类型为2:  '],columns=['预测类型0:  ','预测类型1: ','预测类型2:  '])
print(confusion_matrix_dataframe)                                      #输出混淆矩阵
sum = 0.0                                                              #保存正确率
for i in range(3):
    sum += confusion_matrix[i][i]
print("预测的正确率为：" + str(sum/len(x_test)*100) + '%')               #输出正确率
createPlot(mytree.copy())                                              #画出决策树