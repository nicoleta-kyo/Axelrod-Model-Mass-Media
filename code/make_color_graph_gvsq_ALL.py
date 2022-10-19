# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 23:53:23 2022

@author: nkyos
"""


import numpy as np
import pickle as pkl
from numba import njit
from scipy.ndimage import measurements as meas
import os
import re
import matplotlib.pyplot as plt

#Returns a colored grid
def color(dim, data, L, imageName, save=False):
               
    # turn feature arrays to single values
    Matrix = njit_labelGrid(dim, data, L)
    FlippedMatrix = np.flip(Matrix, 0)
 
    # plot
    fig, ax = plt.subplots() # delte after
    xi = np.arange(0, dim+1)
    yi = np.arange(0, dim+1)
    X, Y = np.meshgrid(xi, yi)
    ax.pcolormesh(X, Y, FlippedMatrix)
    # plt.pcolormesh(FlippedMatrix, edgecolors="w")
    ax.set_aspect('equal')
    plt.axis('off')
    # plt.title('Stable State of Social Influence Model')
    if save:
        plt.savefig(imageName)
    plt.close()
 

@njit
def njit_labelGrid(dim, data, L):
        
    Matrix = np.zeros((dim, dim), dtype = np.uint32)
    labels = np.arange(1, Matrix.size + 1)
    np.random.shuffle(labels)
    
    for row in range (0,dim):
        for col in range(0,dim):
            if row == 0 and col == 0:
                stableRegions = data[row,col].copy().reshape(1, -1)
            else:
                arr_in_regions = False
                for r in range(stableRegions.shape[0]):
                    elems_same = 0
                    for feat in range(stableRegions[r,:].size):
                        if data[row,col,feat] == stableRegions[r, feat]:
                            elems_same += 1
                    if elems_same == L:
                        arr_in_regions = True
                if ~arr_in_regions:
                    stableRegions = np.concatenate((stableRegions, data[row, col].reshape(1, -1)))
                    
    for row in range (0, dim):
        for col in range(0, dim):
            for r in range(stableRegions.shape[0]):
                elems_same = 0
                for feat in range(stableRegions[r,:].size):
                    if data[row,col,feat] == stableRegions[r, feat]:
                        elems_same += 1
                if elems_same == L:
                    Matrix[row,col] = labels[r]
    
    return Matrix


def analyzeClusters(dim, data, L):
     '''
     returns the biggest cluster size (normalised) and the number of clusters
     '''

     
     # turn feature arrays to single values
     Matrix = njit_labelGrid(dim, data, L)
     
     # get number of clusters
     labels = np.unique(Matrix)
     num_clusters = len(labels)
     
     # get cluster sizes 
     clusters_sizes = np.zeros(len(labels), dtype = np.uint32)
     
     for ilab, lab in enumerate(labels):
         z = (Matrix == lab)
         lw, _ = meas.label(z) 
         area = meas.sum(z, lw, index=np.arange(lw.max() + 1))
         clusters_sizes[ilab] = np.max(area)
         
     # get top 1 cluster size
     # N = dim**2
     # norm_clusters_sizes = [i / N for i in clusters_sizes] # normalise sizes
     # g = np.sort(norm_clusters_sizes)[::-1][0]   
     
     g = np.sort(clusters_sizes)[::-1][0]   # normalise after you average!
     
     return g, num_clusters
 

# -- 1    
 

folder = 'Part 1 Results - Runs No3' 
subfolder = '5F50dim1media0.0005B'
path = os.path.join(folder, subfolder)

L = int(re.search('(.*)F', subfolder).group(1))
dim = int(re.search('F(.*)dim', subfolder).group(1))
N = dim*dim
media = int(re.search('dim(.*)media', subfolder).group(1))
B = float(re.search('a(.*)B', subfolder).group(1))
Ts = [2,5,10,15,20,25,30,35,40,45,50]


results = np.zeros((len(Ts), 2)) 

for iT, T in enumerate(Ts):
    
    num_files = 0
    avg_sg = 0
    for folder in os.listdir(path):
        sub_path = os.path.join(path, folder)
        if os.path.isdir(sub_path): 
            for file in os.listdir(sub_path):
                if (('finalMap' in file) and ('.pkl' in file)):
                    if T == int(re.search('F(.*)Q', file).group(1)):
                    
                        print(file)
                        num_files += 1
                        # T = int(re.search('F(.*)Q', file).group(1))
                        
                        filepath = os.path.join(sub_path, file)
                        with open(filepath, 'rb') as f:
                            mat = pkl.load(f)
           
                        _, ng = analyzeClusters(dim, mat, L)
                        
                        imagePath = os.path.join(sub_path, file[:-4].replace('.',','))
                        # color(dim, mat, L, imagePath, save=True)
        
                        avg_sg += ng
               
    mean_q = (avg_sg/num_files) / N if num_files > 0 else 0        
    results[iT, :] =  [T, mean_q] 
    
    
# ----- 2/3/4


folder = 'Part 1 Results - Runs No3' 
# subfolder = '5F50dim2media0.0005B' # missing 5 :(
# subfolder = '5F50dim1media0.6B' # missing one
# subfolder = '5F50dim2media0.1B' # not missing any
path = os.path.join(folder, subfolder)

L = int(re.search('(.*)F', subfolder).group(1))
dim = int(re.search('F(.*)dim', subfolder).group(1))
N = dim*dim
media = int(re.search('dim(.*)media', subfolder).group(1))
B = float(re.search('a(.*)B', subfolder).group(1))
Ts = [2,5,10,15,20,25,30,35,40,45,50]


results = np.zeros((len(Ts), 2)) 

for iT, T in enumerate(Ts):
    
    num_files = 0
    avg_sg = 0
    for folder in os.listdir(path):
        sub_path = os.path.join(path, folder)
        if os.path.isdir(sub_path): 
            for file in os.listdir(sub_path):
                if (('finalMap' in file) and ('.pkl' in file)):
                    if T == int(re.search('F(.*)Q', file).group(1)):
                    
                        print(file)
                        num_files += 1
                        # T = int(re.search('F(.*)Q', file).group(1))
                        
                        filepath = os.path.join(sub_path, file)
                        with open(filepath, 'rb') as f:
                            mat = pkl.load(f)
           
                        _, ng = analyzeClusters(dim, mat, L)
                        
                        imagePath = os.path.join(sub_path, file[:-4].replace('.',','))
                        # color(dim, mat, L, imagePath, save=True)
        
                        avg_sg += ng
               
    mean_q = (avg_sg/num_files) / N if num_files > 0 else 0        
    results[iT, :] =  [T, mean_q]     
    
    
# ------ plot g vs q


figname = path + '.png'
figpath = os.path.join(path, figname)
fig, ax1 = plt.subplots()

color1 = 'black'
lab1 = r'$\langle S_{max} \rangle / N $'
ax1.plot(results[:,0], results[:,1], marker = 'o', color=color1)

ax1.set_xlabel(r'$q$')
ax1.set_ylabel(lab1)
# ax1.tick_params(axis='y')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(figpath, dpi=400)
plt.legend()
plt.show()



                      


