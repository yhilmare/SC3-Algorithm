'''
Created on 2017年10月14日

@author: IL MARE
'''
import csv
import random
import math
import time
import numpy as np
from datetime import datetime as date

initial_classify_num = 3#要划分的类的数目
current_mean_vector_index = list()#当前均值向量的索引
num_digtal = 3#计算值保留小数的位数
resultDict = dict()#最终的分类呈现

def getRandomVectorIndex(dataDict)->None:
    global current_mean_vector_index
    random.seed(date.second)
    randomTemp = set()
    try:
        dictLen = len(dataDict)
        keys = list(dataDict.keys())
        for i in range(0, initial_classify_num):
            temp = random.choice(keys)
            if temp not in randomTemp:
                randomTemp.add(temp)
            else:
                while True:
                    temp = random.choice(keys)
                    if temp not in randomTemp:
                        randomTemp.add(temp)
                        break
            current_mean_vector_index.append((dataDict[temp], i))
    except Exception as e:
        print(e)


# def calDistBetVector(vector1, vector2):
#     if len(vector1) != len(vector2):
#         return -1
#     count = 0
#     for i in range(0, len(vector1)):
#         count += (vector1[i] - vector2[i])**2
#     return math.sqrt(count)
def calDistBetVector(vector1, vector2):
    vec_1 = np.array(vector1, dtype=np.float)
    vec_2 = np.array(vector2, dtype=np.float)
    res = (vec_1 - vec_2)**2
    return np.sqrt(res.sum())
# def calMeanVector(dataDict, vector):
#     result = list(dataDict[vector[0]])
#     for i in range(1, len(vector)):
#         lst = dataDict[vector[i]]
#         for j in range(0, len(lst)):
#             result[j] = result[j] + lst[j]
#     for i in range(0, len(result)):
#         result[i] = round(result[i] / len(vector), num_digtal)
#     return result

def calMeanVector(dataDict, vector):
    result = np.array(dataDict[vector[0]], dtype=np.float)
    for i in range(1, len(vector)):
        lst = np.array(dataDict[vector[i]], dtype=np.float)
        result += lst
    result /= len(vector)
    return list(result)

def convergeToResult(dataDict):
    global current_mean_vector_index, resultDict
    for i in range(0, initial_classify_num):
        resultDict[i] = list()
    for key in dataDict.keys():
        min_count = 10000
        classify_index = -1
        for item in current_mean_vector_index:
            dist = calDistBetVector(dataDict[key], item[0])
            if dist < min_count:
                min_count = dist
                classify_index = item[1]
        resultDict[classify_index].append(key)
    print("=====" * 8)
    tempList = list()
    for item in resultDict.items():
        tempList.append((calMeanVector(dataDict, item[1]), item[0]))
    if current_mean_vector_index != tempList:
        current_mean_vector_index = tempList
        return -1
    else:
        return 1


def wirteToFile(destDir):
    try:
        global resultDict
        fp = open(r'%s%.3f.csv' % (destDir, time.clock()), 'w', newline='\n')
    except Exception as e:
        print(e)
    else:
        writer = csv.writer(fp)
        for items in resultDict.items():
            for value in items[1]:
                writer.writerow([value, str(items[0])])
    finally:
        fp.close()

def run_kmeans_cluster(samples, classify_num = 3, digtal_num = 3):
    global initial_classify_num, num_digtal
    initial_classify_num = classify_num
    num_digtal = digtal_num
    getRandomVectorIndex(samples)  # 初始化均值向量的索引号
    while True:
        if convergeToResult(samples) == 1:
            break
    return resultDict

if __name__ == "__main__":
    pass