# -*- coding: utf-8 -*-
"""
Created on Mon May 30 18:56:31 2022

@author: nkyos
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def color(height,width,data,imageName):
     H = height
     W = width
     Matrix = np.zeros((H, W))
     stableRegions=[]
     for row in range (0,H):
         for col in range(0,W):
             if data[row][col] not in stableRegions:
                 stableRegions.append(data[row][col])
     for row in range (0,H):
         for col in range(0,W):
             for i in range (len(stableRegions)):
                 if data[row][col]==stableRegions[i]:
                     Matrix[row,col]=i+1   
     Matrix=np.flip(Matrix,0)
     xi = np.arange(0, W+1)
     yi = np.arange(0, H+1)
     X, Y = np.meshgrid(xi, yi)
     plt.pcolormesh(X, Y, Matrix)
     plt.pcolormesh(Matrix, edgecolors="w")
     plt.axis('off')
     plt.title('Stable State of Social Influence Model')
     plt.savefig(imageName)
     plt.close()
     
#

# path = 'F=5_q=10_dim=50_B=2_m=1/'
# filename = '0map5F30Q50dim1mediaNonemess0.0005B.png'
# fpath = path + filename + '.pkl'

    
# with open(fpath, 'rb') as file:
#     res = pickle.load(file)
    
    
# height = width = 50
# imageName = path[:-1] +'.png' # !!!!!!!!! change for initial pic
# color(height, width, res, path+imageName)


# iterate through files

path = 'F=5_q=10_dim=50_B=0.01_m=2/'
for file in os.listdir(path):
    if '.pkl' in file:
        filepath = os.path.join(path, file)
        with open(filepath, 'rb') as f:
            res = pickle.load(f)
        height = width = 50
        imagePath = filepath[:-8] +'.png'
        color(height, width, res, imagePath)