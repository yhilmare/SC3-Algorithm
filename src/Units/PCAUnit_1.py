'''
Created on 2017年11月5日

@author: IL MARE
'''
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PCA.PCAUtil as pca
import kmeanslib.KMeansUtil as kmeans

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

def read_files(filename=r'g:/ML/temp/result.csv'):
    try:
        fp = open(filename, 'r')
        reader = csv.reader(fp)
        result = dict()
        for item in reader:
            result[str(item[0])] = item[1:]
        return result
    except Exception as e:
        print(e)
    finally:
        fp.close()

if __name__ == '__main__':
    pass