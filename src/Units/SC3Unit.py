'''
Created on 2017年11月1日

@author: IL MARE
'''
import csv
import numpy as np
import math
import PCA.PCAUtil as pca
import time
import matplotlib as mpl
from matplotlib import pyplot as plt
from datetime import datetime as date
import os

filter_index = 0.06
def read_data_from_disk(filename=r'g:/ML/temp/filter_gene_1.csv'):#对原始数据求对数，轻易不要调用
    try:
        fp = open(filename, 'r')
        reader = csv.reader(fp)
        res = []
        for item in reader:
            temp = []
            temp.append(item[0])
            for i in range(1, len(item)):
                temp.append(math.log(float(item[i]) + 1, 2))
            res.append(temp)
        fp1 = open(r'g:/ML/temp/filter_gene_2.csv', 'w', newline='\n')
        writer = csv.writer(fp1)
        for item in res:
            writer.writerow(item)
    except Exception as e:
        print(e)
    finally:
        fp.close()
        fp1.close()

def cal_pearson_dis(vector1, vector2):#计算两个向量之间的皮尔森距离
    if len(vector1) != len(vector2):
        return -1
    temp_vec_1 = np.array(vector1, dtype=np.float)
    temp_vec_2 = np.array(vector2, dtype=np.float)
    temp_vec_1_mean = temp_vec_1.mean()
    temp_vec_2_mean = temp_vec_2.mean()
    factor_1 = 0
    for num_1, num_2 in zip(temp_vec_1, temp_vec_2):
        factor_1 += (num_1 - temp_vec_1_mean) * (num_2 - temp_vec_2_mean)
    factor_2 = 0
    for num in temp_vec_1:
        factor_2 += (num - temp_vec_1_mean) ** 2
    factor_3 = 0
    for num in temp_vec_2:
        factor_3 += (num - temp_vec_2_mean) ** 2
    factor_2 = math.sqrt(factor_2 * factor_3)
    return factor_1 / factor_2

def calDistBetVector(vector1, vector2):#计算两个向量之间的欧式距离
    if len(vector1) != len(vector2):
        return -1
    count = 0
    for i in range(0, len(vector1)):
        count += (float(vector1[i]) - float(vector2[i]))**2
    return math.sqrt(count)

def cal_distance_matrix(func, filename=r'g:/ML/temp/filter_gene_2.csv'):
    try:
        fp = open(filename, 'r')
        reader = csv.reader(fp)
        temp_dict = dict()
        for cell in reader:
            temp_dict[cell[0]] = cell[1:]
        res = []
        temp_lst = []
        temp_lst.append("")
        for key in temp_dict.keys():
            temp_lst.append(key)
        res.append(temp_lst)
        for key_1 in temp_dict.keys():
            print(key_1)
            temp_lst_1 = []
            temp_lst_1.append(key_1)
            for key_2 in temp_dict.keys():
                temp_lst_1.append(func(temp_dict[key_1], temp_dict[key_2]))
            res.append(temp_lst_1)
        # fp1 = open(r'/Users/yh_swjtu/Desktop/temp/gene_matrix_pearson.csv', 'w', newline='\n')
        # writer = csv.writer(fp1)
        # for item in res:
        #     writer.writerow(item)
    except Exception as e:
        print(e)
    finally:
        fp.close()
        # fp1.close()

def read_distance_matrix_from_disk(filename=r'g:/ML/temp/gene_matrix_euclidean.csv'):
    try:
        fp = open(filename, 'r')
        reader = csv.reader(fp)
        yield reader.__next__()
        result = []
        for item in reader:
            result.append(item[1:])
        yield np.array(result, dtype=np.float)
    except Exception as e:
        print(e)
    finally:
        fp.close()


if __name__ == '__main__':
    try:
        fp = open(r'g:/ML/temp/备份数据/pearson_conment_matrix.csv', 'r')
        fp1 = open(r'g:/ML/temp/备份数据/euclidean_conment_matrix.csv', 'r')
        reader = csv.reader(fp)
        reader1 = csv.reader(fp1)
        index_lst = reader.__next__()
        reader1.__next__()
        matrix_1 = []
        matrix_2 = []
        for row in reader:
            matrix_1.append(row[1:])
        for row in reader1:
            matrix_2.append(row[1:])
        matrix_1 = np.array(matrix_1, dtype=np.float)
        matrix_2 = np.array(matrix_2, dtype=np.float)
        matrix_1 = (matrix_1 + matrix_2) / 2
        fp3 = open(r'g:/ML/temp/finally_conment_matrix.csv', 'w', newline='\n')
        writer = csv.writer(fp3)
        writer.writerow(index_lst)
        for item in enumerate(matrix_1):
            writer.writerow([index_lst[item[0] + 1], *item[1]])
    except Exception as e:
        print(e)
    finally:
        fp.close()
        fp1.close()
        fp3.close()
#     dirPath = r'g:/ML/temp/euclidean_matrix/'
#     count_matrix = np.zeros((420, 420), dtype=np.float)
#     names = os.listdir(dirPath)
#     index_lst = []
#     for filename in names:
#         filePath = '{0}{1}'.format(dirPath, filename)
#         try:
#             temp_matrix = []
#             fp = open(filePath, 'r')
#             reader = csv.reader(fp)
#             index_lst = reader.__next__()
#             for row  in reader:
#                 temp_matrix.append(row[1:])
#             temp_matrix = np.array(temp_matrix, dtype = np.float)
#             count_matrix = count_matrix + temp_matrix
#         except Exception as e:
#             print(e)
#         finally:
#             fp.close()
#     count_matrix = count_matrix / len(names)
#     try:
#         fp = open(r'g:/ML/temp/euclidean_conment_matrix.csv', 'w', newline='\n')
#         writer = csv.writer(fp)
#         writer.writerow(index_lst)
#         for item in enumerate(count_matrix):
#             writer.writerow([index_lst[item[0] + 1], *item[1]])
#     except Exception as e:
#         print(e)
#     finally:
#         fp.close()
    #===========================以下代码用PCA对距离矩阵进行降维，注意路径别写错了=========================================
#     for i in range(2, 3):
#         print("Running iteration %d" % (i), date.now(), sep='  ')
#         start = time.clock()
#         func = iter(read_distance_matrix_from_disk())
#         name_index = func.__next__()
#         samples = func.__next__()
#         pca_matrix = pca.reduce_the_dim_matrix(samples, i)
#         try:
#             fp = open(r'g:/ML/temp/pca_euclidean/pca_{0}dim_matrix_for_euclidean.csv'.format(i), 'w', newline='\n')
#             writer = csv.writer(fp)
#             for name, row in zip(name_index[1:], pca_matrix):
#                 writer.writerow([name, *row])
#         except Exception as e:
#             print(e)
#         finally:
#             fp.close()
#         print('The program run {0}s'.format(round((time.clock() - start), 3)), date.now(), sep='  ')
    #===============================以下代码实现距离矩阵的计算===================================
    #cal_distance_matrix(func=calDistBetVector)
    #read_data_from_disk()