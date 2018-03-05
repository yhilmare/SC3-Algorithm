'''
Created on 2017年11月1日

@author: IL MARE
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def func():
    try:
        fp = open(r'G:\研究生课件\机器学习\ML\consensus.csv', 'r')
        reader = csv.reader(fp)
        #x = reader.__next__()[1:]
        matrix = []
        for row in reader:
            matrix.append(row)
    except Exception as e:
        print(e)
    else:
        matrix = np.array(matrix, dtype = np.float)
        print(matrix.shape)
        x = np.arange(420)
        x, y = np.meshgrid(x, x)
        mpl.rcParams['xtick.labelsize'] = 6
        mpl.rcParams['ytick.labelsize'] = 6
        fig = plt.figure("Test Data", figsize=(7,7))
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.view_init(elev=87, azim=-120)
        ax.plot_surface(x, y, matrix, cmap=cm.coolwarm, rstride=10, cstride=10, lw=0)
        plt.show()
    finally:
        fp.close()

if __name__ == '__main__':
    func()
#     try:
#         fp = open(r'g:/ML/consensus.txt', 'r')
#         matrix = []
#         fp.__next__()
#         for item in fp:
#             temp = []
#             iter = item.split(' ')
#             for i in range(1, len(iter)):
#                 temp.append(float(iter[i]))
#             matrix.append(temp)
#     except Exception as e:
#         print(e)
#     else:
#         matrix = np.array(matrix, dtype = np.float)
#         x = np.arange(420)
#         x, y = np.meshgrid(x, x)
#         mpl.rcParams['xtick.labelsize'] = 6
#         mpl.rcParams['ytick.labelsize'] = 6
#         fig = plt.figure("Test Data", figsize=(7,7))
#         ax = fig.add_subplot(1,1,1, projection='3d')
#         ax.view_init(elev=87, azim=-120)
#         ax.plot_surface(x, y, matrix, cmap=cm.coolwarm, rstride=10, cstride=10, lw=1)
#         plt.show()
#     finally:
#         fp.close()        