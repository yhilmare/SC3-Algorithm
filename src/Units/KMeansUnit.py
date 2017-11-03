'''
Created on 2017年11月1日

@author: IL MARE
'''
import kmeanslib.KMeansUtil as km
import csv
import PCA.PCAUtil as pa
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from lib2to3.fixer_util import Newline

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

# def getDataDict(fileName:"to input the filename"="/Users/yh_swjtu/Desktop/scRNAseq_CountTable.csv")->dict:
#     try:
#         fp = open(fileName)
#     except Exception as e:
#         print(e)
#         return None
#     else:
#         reader = csv.reader(fp)
#         dic = dict()
#         for item in reader.__next__():
#             dic[item] = list()
#         i = 0
#         for item in reader:
#             k = 0
#             for key in dic.keys():
#                 dic[key].append(float(item[k]))
#                 k = k + 1
#             print(i)
#             i += 1
#         return dic
#     finally:
#         fp.close()

def read_samples_from_disk_forPCA(filename=r'g:/ML/temp/result.csv'):
    try:
        fp = open(filename, 'r')
        result = []
        reader = csv.reader(fp)
        for item in reader:
            result.append(item[1:])
        return np.array(result, dtype=np.float)
    except Exception as e:
        print(e)
    finally:
        fp.close()


def parse_data_for_kMeans(samples):
    result = dict()
    for item in enumerate(samples):
        result[str(item[0] + 1)] = list(item[1])
    return result

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

if __name__ == '__main__':
#     samples = read_samples_from_disk_forPCA()#从文件中为pca降维读到数据
#     dest_matrix = pa.reduce_the_dim_matrix(samples)#利用pca得到降维后的数据
#     pca_data = parse_data_for_kMeans(dest_matrix)#将降维后的数据进行处理，得到kmeans能用的数据
#==========================对距离矩阵进行kmeans聚类，注意路径问题================================================
    try:
        fp = open(r'g:/ML/temp/filter_gene_2.csv', 'r')
    except Exception as e:
        print(e)
    else:
        index_lst = []
        reader = csv.reader(fp)
        for row in reader:
            index_lst.append(row[0])
        i = 30
        pca_data = getDataDict(fileName=r'g:/ML/temp/pca_euclidean/pca_{0}dim_matrix_for_euclidean.csv'.format(i))
        kmeans_matrix = km.run_kmeans_cluster(pca_data, 9)#进行kmeans聚类得到结果
        wirte_to_file(r'g:/ML/temp/euclidean_matrix/pca_{0}dim_matrix_for_euclidean.csv'.format(i), kmeans_matrix, index_lst)
    finally:
        fp.close()
#     total_res = []
#     for item in kmeans_matrix.items():
#         res = []
#         for index_lst in item[1]:
#             res.append(pca_data[index_lst])
#         total_res.append(res)
#     mpl.rcParams['xtick.labelsize'] = 6
#     mpl.rcParams['ytick.labelsize'] = 6
#     fig = plt.figure('Test Data')
#     ax = fig.add_subplot(111)
#     ax.set_xlabel('dim1')
#     ax.set_ylabel('dim2')
#     # ax.set_xlim([0, 10])
#     # ax.set_ylim([0, 8])
#     for sub_fig in total_res:
#         dest = np.array(sub_fig)
#         ax.plot(dest[:, 0], dest[:, 1], '*')
#     plt.show()
