'''
Created on 2017年11月1日

@author: IL MARE
'''
import csv
import math
import os

def getResultDict(fileName=r'/Users/yh_swjtu/Desktop/机器学习（分类）/数据标准分类结果/result.csv'):
    try:
        fp = open(fileName, 'r')
        reader = csv.reader(fp)
        resultDict = dict()
        for item in reader:
            if resultDict.get(item[1]) == None:
                resultDict[item[1]] = list()
            resultDict[item[1]].append(item[0])
        return resultDict
    except Exception as e:
        print(e)
        return None
    finally:
        fp.close()
def getClassifyResult(instName, resultDict):
    for item in resultDict.items():
        if instName in item[1]:
            return item[0]
def getClassifyTest(instName, testDict):
    for item in testDict.items():
        if instName in item[1]:
            return item[0]

def scanAllTheListA(lst, resultDict, testDict):
    result = set()
    for i in range(0, len(lst)):
        inst1 = lst[i]
        for j in range(i + 1, len(lst)):
            inst2 = lst[j]
            if getClassifyResult(inst1, resultDict) == getClassifyResult(inst2, resultDict) and getClassifyTest(inst1, testDict) == getClassifyTest(inst2, testDict):
                result.add((inst1, inst2))
    return result
def scanAllTheListB(lst, resultDict, testDict):
    result = set()
    for i in range(0, len(lst)):
        inst1 = lst[i]
        for j in range(i + 1, len(lst)):
            inst2 = lst[j]
            if getClassifyResult(inst1, resultDict) == getClassifyResult(inst2, resultDict) and getClassifyTest(inst1, testDict) != getClassifyTest(inst2, testDict):
                result.add((inst1, inst2))
    return result
def scanAllTheListC(lst, resultDict, testDict):
    result = set()
    for i in range(0, len(lst)):
        inst1 = lst[i]
        for j in range(i + 1, len(lst)):
            inst2 = lst[j]
            if getClassifyResult(inst1, resultDict) != getClassifyResult(inst2, resultDict) and getClassifyTest(inst1, testDict) == getClassifyTest(inst2, testDict):
                result.add((inst1, inst2))
    return result
def scanAllTheListD(lst, resultDict, testDict):
    result = set()
    for i in range(0, len(lst)):
        inst1 = lst[i]
        for j in range(i + 1, len(lst)):
            inst2 = lst[j]
            if getClassifyResult(inst1, resultDict) != getClassifyResult(inst2, resultDict) and getClassifyTest(inst1, testDict) != getClassifyTest(inst2, testDict):
                result.add((inst1, inst2))
    return result
if __name__ == "__main__":
    resultCSV = []
    for filename in os.listdir(r'/Users/yh_swjtu/Desktop/机器学习（分类）/lalala/'):
        testDict = getResultDict('/Users/yh_swjtu/Desktop/机器学习（分类）/lalala/%s' % (filename))
        resultDict = getResultDict()
        testResult = []
        for items in testDict.items():
            for item in items[1]:
                testResult.append(item)
        A = scanAllTheListA(testResult, resultDict, testDict)
        B = scanAllTheListB(testResult, resultDict, testDict)
        C = scanAllTheListC(testResult, resultDict, testDict)
        D = scanAllTheListD(testResult, resultDict, testDict)
        JC = round(len(A)/(len(A) + len(B) + len(C)), 3)
        print('JC index is:', JC)
        temp = (len(A)/(len(A) + len(B)))*(len(A)/(len(A) + len(C)))
        FML = round(math.sqrt(temp), 3)
        print('FMI index is:', FML)
        RI = round(2*(len(A) + len(D))/(420*419), 3)
        print('RI index is:', RI)
        resultCSV.append([filename[0:-4], JC, FML, RI])
        print("========" * 4, filename, '========' * 4)
    try:
        fp = open(r'/Users/yh_swjtu/Desktop/机器学习（分类）/lalala/result1.csv', 'w', newline='\n')
        writer = csv.writer(fp)
    except Exception as e:
        print(e)
    else:
        writer.writerow(['文件名', 'JC指数', 'FMI指数', 'RI指数'])
        for items in resultCSV:
            writer.writerow(items)
    finally:
        fp.close()