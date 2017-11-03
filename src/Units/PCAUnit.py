'''
Created on 2017年11月1日

@author: IL MARE
'''
import PCA.PCAUtil as pca
import csv
import numpy as np
import re 

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

if __name__ == '__main__':
    samples = read_file_from_disk()
    print(pca.reduce_the_dim_matrix(samples))