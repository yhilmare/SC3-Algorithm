'''
Created on 2017年11月1日

@author: IL MARE
'''
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy

print('import package {0}'.format(__name__))
destination_dim = 2
def read_file_from_disk(filename=r'g:/ML/temp/result.csv'):
    try:
        fp = open(filename, 'r')
        reader = csv.reader(fp)
        result = []
        for item in reader:
            result.append(item[1:])
        return np.array(result, dtype=np.float)
    except Exception as e:
        print(e)
        return None
    finally:
        fp.close()

def centralized_samples(samples):
    temp = np.ones((1, samples.shape[1]), dtype=np.float)[0]
    for item in samples:
        temp += item
    temp /= samples.shape[0]
    for item in samples:
        item -= temp
    return samples


def calculate_cov(samples):
    horizontal_dim = samples.shape[0]
    vertical_dim = samples.shape[1]
    result_mean = []
    cov_matrix = []
    for i in range(0, vertical_dim):
        result_mean.append(samples[:,i].mean())
    for i in range(0, vertical_dim):
        temp_list = []
        for j in range(0, vertical_dim):
            count = 0
            for k in range(0, horizontal_dim):
                count += (samples[k][i] - result_mean[i]) * (samples[k][j] - result_mean[j])
            count /= (horizontal_dim - 1)
            temp_list.append(count)
        cov_matrix.append(temp_list)
    return np.array(cov_matrix)

def shadow_new_coodinary(eig_value, eig_vector):
    shadow_matrix = []
    temp_list = []
    index_list = []
    for i in range(0, destination_dim):
        temp_count = 0
        for value in eig_value:
            if value not in temp_list:
                temp_count = value if value > temp_count else temp_count
        temp_list.append(temp_count)
        for item in enumerate(eig_value):
            if temp_count == item[1]:
                index_list.append(item[0])
                break
    for i in range(0, destination_dim):
        shadow_matrix.append(list(eig_vector[:,index_list[i]]))
    return np.array(shadow_matrix)

def reduce_the_dim_matrix(samples, dest_dim = 2):
    global destination_dim
    destination_dim = dest_dim
    sec_samples = copy.deepcopy(samples)
    centralized_samples(samples)  # 进行中心化后的数据
    cov_matrix = calculate_cov(samples)  # 计算中心化数据的协方差，得到协方差矩阵
    eig_value, eig_vector = np.linalg.eigh(cov_matrix)  # 计算协方差矩阵的特征值和特征向量
    shadow_matrix = shadow_new_coodinary(eig_value, eig_vector)  # 得到投影矩阵
    samples = np.dot(sec_samples, shadow_matrix.T)  # 得到降维后的矩阵
    return samples

if __name__ == '__main__':
    samples = read_file_from_disk()#初始的数据
    centralized_samples(samples)#进行中心化后的数据
    cov_matrix = calculate_cov(samples)#计算中心化数据的协方差，得到协方差矩阵
    eig_value, eig_vector = np.linalg.eig(cov_matrix)#计算协方差矩阵的特征值和特征向量
    shadow_matrix = shadow_new_coodinary(eig_value, eig_vector)#得到投影矩阵
    samples = np.dot(read_file_from_disk(), shadow_matrix.T)#得到降维后的矩阵
    print(samples)
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    fig = plt.figure('Test Data')
    ax = fig.add_subplot(111)
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 8])
    ax.plot(samples[:,0], np.abs(samples[:,1]), '*')
    plt.show()