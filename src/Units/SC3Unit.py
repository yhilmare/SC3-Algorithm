'''
Created on 2017年11月1日

@author: IL MARE
'''
import csv
import numpy as np
import math
import time
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from datetime import datetime as date
sys.path.append(os.getcwd())
import kmeanslib.KMeansUtil as km
import PCA.PCAUtil as pca

filter_index = 0.06

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

def cal_euclidean_dis(vector1, vector2):#计算两个向量之间的欧式距离
    if len(vector1) != len(vector2):
        return -1
    count = 0
    for i in range(0, len(vector1)):
        count += (float(vector1[i]) - float(vector2[i]))**2
    return math.sqrt(count)

def cal_spearman_dis(vector1, vector2):
    if len(vector1) != len(vector2):
        return -1
    count = 0
    for i in range(0, len(vector1)):
        count += (float(vector1[i]) - float(vector2[i]))**2
    n = len(vector1)
    return 1 - (6 * count) / (n * (n**2 - 1))

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
            temp_lst_1 = []
            temp_lst_1.append(key_1)
            for key_2 in temp_dict.keys():
                temp_lst_1.append(func(temp_dict[key_1], temp_dict[key_2]))
            res.append(temp_lst_1)
        return res
    except Exception as e:
        print(e)
    finally:
        fp.close()

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

def getDataDict(fileName:"to input the filename"="g:/ML/temp/temp.csv")->dict:
    try:
        fp = open(fileName)
    except Exception as e:
        print(e)
        return None
    else:
        reader = csv.reader(fp)
        dic = dict()
        for item in reader:
            tempList = list()
            for i in range(1, len(item)):
                tempList.append(float(item[i]))
            dic[item[0]] = tempList
        return dic
    finally:
        fp.close()

def get_the_classify_result(index, result_matrix):
    for item in result_matrix.items():
        if index in item[1]:
            return item[0]
    return -1

def wirte_to_file(filename, result_matrix, index_lst):
    try:
        fp = open(filename, 'w', newline='\n')
        writer = csv.writer(fp)
        result_lst = []
        temp_lst = []
        temp_lst.append('')
        for item in index_lst:
            temp_lst.append(item)
        result_lst.append(temp_lst)
        for i in range(0, len(index_lst)):
            temp = []
            temp.append(index_lst[i])
            for j in range(0, len(index_lst)):
                index_1 = get_the_classify_result(index_lst[i], result_matrix)
                index_2 = get_the_classify_result(index_lst[j], result_matrix)
                if index_1 == index_2:
                    temp.append('1')
                else:
                    temp.append('0')
            result_lst.append(temp)
        for item in result_lst:
            writer.writerow(item)     
    except Exception as e:
        print(e)
    finally:
        fp.close()

def read_matrix_from_file(filename):
    try:
        fp = open(filename)
        reader = csv.reader(fp)
        res = []
        reader.__next__()
        for item in reader:
           res.append(item[1:])
        return np.array(res, dtype=np.float)
    except Exception as e:
        print(e)
        return None
    finally:
        fp.close()


def plot_func(matrix):
    x = np.arange(420)
    x, y = np.meshgrid(x, x)
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    fig = plt.figure("Test Data", figsize=(7,7))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.view_init(elev=87, azim=-120)
    ax.plot_surface(x, y, matrix, cmap=cm.coolwarm, rstride=1, cstride=1, lw=0)
    plt.show()

def read_data_from_disk(cwd, data_set):#对原始数据求对数，轻易不要调用
    try:
        fp = open(cwd + data_set, 'r')
        reader = csv.reader(fp)
        tmp_lst = []
        name_lst = reader.__next__()[1:]
        total_num = len(name_lst)
        for item in reader:
            count = 0
            for num in item[1:]:
                if float(num) != 0:
                    count += 1
            tmp = count / total_num
            if tmp > filter_index and tmp < (1 - filter_index):
                tmp_lst.append(item[1:])
        res = np.array(tmp_lst, dtype=np.float)
        res = res.T
        fp1 = open(cwd + "filter_gene_2.csv", "w", newline="\n")
        writer = csv.writer(fp1)
        for item in zip(name_lst, res):
            tmp_list = []
            tmp_list.append(item[0])
            for i in item[1]:
                tmp_list.append(math.log(float(i) + 1, 2))
            writer.writerow(tmp_list)
    except Exception as e:
        print(e)
    finally:
        fp.close()
        fp1.close()

if __name__ == '__main__':
    start = time.clock()
    cwd = os.getcwd() + "\\"
    data_set = "scRNAseq_CountTable.csv"
    read_data_from_disk(cwd, data_set)
    filter_gene = cwd + "filter_gene_2.csv"
    func_lst = [cal_euclidean_dis, cal_pearson_dis, cal_spearman_dis]
    dir_paths = []
    print("=" * 4, "正在计算距离矩阵， 当前处理文件...", filter_gene, "=" * 4)
    for func in func_lst:
        print("调用函数{0}...".format(func.__name__))
        if not os.path.exists(cwd + func.__name__.split("_")[1]):
            os.mkdir(cwd + func.__name__.split("_")[1])
        dir_paths.append(cwd + func.__name__.split("_")[1])
        cur_dir = cwd + func.__name__.split("_")[1] + "/"
        try:
            fp = open(cur_dir + "{0}_distance_matrix.csv".format(func.__name__.split("_")[1]), "w", newline="\n")
            writer = csv.writer(fp)
            for item in cal_distance_matrix(func, filter_gene):
                writer.writerow(item)
        except Exception as e:
            print(e)
        finally:
            fp.close()
    stamp_1 = time.clock() - start
    print("=" * 4, "距离矩阵计算完毕，用时{0:.3f}秒".format(stamp_1), "=" * 4)
    print("=" * 4, "开始对距离矩阵进行降维...", "=" * 4)
    for dir in dir_paths:
        tmp_name =dir.split("\\")[-1]
        file_name = dir + "\\" + "{0}_distance_matrix.csv".format(tmp_name)
        print("当前处理的距离矩阵为", file_name, "...")
        func = iter(read_distance_matrix_from_disk(file_name))
        name_index = func.__next__()
        samples = func.__next__()
        for i in range(17, 31):
            print("对{0}矩阵降维，目标维数{1}维...".format(tmp_name, i))
            pca_matrix = pca.reduce_the_dim_matrix(samples, i)
            try:
                fp = open(r'{0}/pca_{1}dim_matrix_for_{2}.csv'.format(dir, i, tmp_name), 'w', newline='\n')
                writer = csv.writer(fp)
                for name, row in zip(name_index[1:], pca_matrix):
                    writer.writerow([name, *row])
            except Exception as e:
                print(e)
            finally:
                fp.close()
    stamp_2 = time.clock() - stamp_1
    print("=" * 4, "距离矩阵降维计算完毕，用时{0:.3f}秒".format(stamp_2), "=" * 4)
    print("=" * 4, "开始对距离矩阵进行KMeans聚类...", "=" * 4)
    matrix_path = []
    try:
        fp = open(filter_gene, 'r')
        index_lst = []
        reader = csv.reader(fp)
        for row in reader:
            index_lst.append(row[0])
        for dir in dir_paths:
            tmp_name =dir.split("\\")[-1]
            print("正在对{0}距离矩阵进行KMeans聚类...".format(tmp_name))
            matrix_path.append(cwd + "{0}_matrix".format(tmp_name))
            for i in range(17, 31):
                print("******正在对{0}距离矩阵的{1}维数据进行聚类...".format(tmp_name, i))
                file_name = r'{0}/pca_{1}dim_matrix_for_{2}.csv'.format(dir, i, tmp_name)
                pca_data = getDataDict(file_name)
                kmeans_matrix = km.run_kmeans_cluster(pca_data, classify_num = 9)#进行kmeans聚类得到结果
                if not os.path.exists(cwd + "{0}_matrix".format(tmp_name)):
                    os.mkdir(cwd + "{0}_matrix".format(tmp_name))
                wirte_to_file(r'{0}/KMeans_consensus_for_{1}dim_{2}_matrix.csv'.format(cwd + "{0}_matrix".format(tmp_name), i, tmp_name), kmeans_matrix, index_lst)
    except Exception as e:
        print(e)
    finally:
        fp.close()
    stamp_3 = time.clock() - stamp_2 - stamp_1
    print("=" * 4, "距离矩阵KMeans聚类共识矩阵计算完毕，用时{0:.3f}秒".format(stamp_3), "=" * 4)
    print("=" * 4, "计算整体共识矩阵...", "=" * 4)
    count_matrix = np.zeros((420, 420), dtype=np.float)
    count = 0
    for item in matrix_path:
        for filename in os.listdir(item):
            count += 1
            count_matrix += read_matrix_from_file(item + "/" + filename)
    count_matrix /= count
    try:
        fp = open(r'{0}\finally_conment_matrix.csv'.format(cwd), 'w', newline='\n')
        writer = csv.writer(fp)
        for row in count_matrix:
            writer.writerow(row)
    except Exception as e:
        print(e)
    finally:
        fp.close()
    stamp_4 = time.clock() - stamp_3 - stamp_2 - stamp_1
    print("=" * 4, "共识矩阵计算完成，耗时{0:.3f}秒".format(stamp_4), "=" * 4)
    plot_func(count_matrix)
    print("=" * 4, "程序运行完成，耗时{0:.3f}秒".format(time.clock() - start), "=" * 4)