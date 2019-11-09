



#导入相关库函数
import pandas as pd

#得到数据以嵌套链表的形式读入
def createData():
    dataSet_dataframe = pd.read_excel('./apriori.xlsx')
    dataSet = []
    for row in dataSet_dataframe.itertuples():
        row_temp = list(row)
        del row_temp[0:2]
        del row_temp[-1]
        dataSet.append(row_temp)
    return dataSet

## 先得到频繁集

def toSingleSet(data):    # 得到单个[frozenset(item)]集合
    singleSet = []
    for transaction in data:
        for item in transaction:
            if [item] not in singleSet:
                singleSet.append([item])
    return list(map(frozenset,singleSet))

def scan_data(data_set,data_simple,minsupport):
    fk_support_temp = {}
    for trasaction in data_set:
        for temp in data_simple:
            if temp.issubset(trasaction):
                if temp not in fk_support_temp:
                    fk_support_temp[temp] = 1
                else:
                    fk_support_temp[temp] += 1
    fk_support = {}
    fk = []
    num = float(len(data_set))
    for key in fk_support_temp:
        support_temp = fk_support_temp[key] / num
        if support_temp >= minsupport:
            fk.append(key)
        fk_support[key] = support_temp
    return fk,fk_support

def apriori_gen(FK,k):
    FK_NEXT = []
    for i in range(len(FK)):
        for j in range(i+1,len(FK)):
            temp = FK[i]|FK[j]
            if len(temp) == k:
                if temp not in FK_NEXT:
                    FK_NEXT.append(temp)
    return FK_NEXT


def apriori(data,minSupport=0.3):
    data_set = list(map(set,data))    # 将data数据元素每个transaction都变成集合
    data_f1 = toSingleSet(data)   # 返回该数据集中所有item元素组成的frozeset链表
    FK,FK_suport =  scan_data(data_set,data_f1,minSupport) #扫描数据，计算执行度，去除不符合置信度的集合，得到频繁集和对应频繁集：支持度
    result = [FK]
    k = 2
    while len(result[k -2])>0:     #频繁集集合个数是大于0的
        Ck = apriori_gen(result[k-2],k)
        FK,FK_suport_temp = scan_data(data_set,Ck,minSupport)
        result.append(FK)
        FK_suport.update(FK_suport_temp)
        k += 1
    print(result)
    print(FK_suport)
    return result,FK_suport

##################################################################################################################


##################################################################################################################
##挖掘频繁集，得到规则

def toGetRules(results,FK_support,mincof = 0.2):  #获得所有频繁子集的关联规则
    rules = []
    for i in range(1,len(result)):
        for fk in result[i]:
            data_consequence = [frozenset([item]) for item in fk]
            if(i > 1):  # 如何i>1 进行进一步处理得到所有的后件即结论
                toGet_single_conseqense(fk,data_consequence,FK_support,mincof,rules)
            else:
                cacal_conf(fk,data_consequence,FK_support,mincof,rules)
    return  rules

def cacal_conf(fk,temp_conse,FK_support,minconf,rules):  #计算置信度，得到符合置信度的规则集合
    new_conse = []
    for item in temp_conse:
        conf = FK_support[fk]/FK_support[fk-item]
        if conf >= minconf:
            new_conse.append(temp_conse)
            rules.append((fk-item,item,conf))
            print('{:>40}'.format(str(fk-item)),'------>',item)
    return new_conse


def toGet_single_conseqense(fk,data_consequence,FK_support,minconf,rules):  #得到单个频繁集的所有后项集
    temp_len = len(data_consequence[0])
    if(len(fk)>=(temp_len+1)):
        temp_conse = cacal_conf(fk, data_consequence, FK_support, minconf, rules)
        temp_conse = apriori_gen(data_consequence,temp_len+1)
        if len(temp_conse)>1:
            toGet_single_conseqense(fk,temp_conse,FK_support,minconf,rules)
##################################################################################################################



##################################################################################################################
## 主函数调用部分
data  = createData()
result,FK_support=apriori(data,0.3)
rules = toGetRules(result,FK_support,0.60)
print(rules)
