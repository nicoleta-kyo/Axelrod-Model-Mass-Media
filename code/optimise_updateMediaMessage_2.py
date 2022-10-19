# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:11:31 2022

@author: nkyos
"""
import numpy as np
from numpy.random import PCG64
from numba import njit
from timeit import timeit


# my usual rand
rng = np.random.default_rng(12345)


@njit
def createRandom_np():
    
    '''
    numpy array instead of list of lists
    values are integers, not strings
    
    old random numpy
    '''
    
    data = np.zeros((H,W,L),dtype=np.uint32)
    neighbors = np.zeros((H,W,4,2), dtype=np.uint32) # 4 neighbors, row and col position
    # neighbors[:,:] = -1

    K = np.arange(T)
        
    for row in range(0,H):
        for col in range(0,W):
            neighbors[row][col] = neighborhood_np(row, col)
            inds_choices = np.random.randint(0, T, L)
            data[row][col] = K[inds_choices]
            
    return data, neighbors
       
@njit
def neighborhood_np(row, col):
       '''
       not removing out of range neighbors
       '''

       neighborhood = np.zeros((4,2), dtype = np.uint32)
       neighborhood[0,:] = np.array([row+1,col], dtype = np.uint32)
       neighborhood[1,:] = np.array([row-1,col], dtype = np.uint32)
       neighborhood[2,:] = np.array([row,col+1], dtype = np.uint32)
       neighborhood[3,:] = np.array([row,col-1], dtype = np.uint32)
       
       # # remove out of range neighbors
       # inds1 = (neighborhood >= 0)[:,0].nonzero()[0]
       # inds2 = (neighborhood < H)[:,1].nonzero()[0]
       # neighborhood = neighborhood[np.intersect1d(inds1, inds2),:]   
           
       return neighborhood


def updateMediaMessage(rng):
    
    media_message = np.zeros((H,W,L), dtype = np.uint32)
    
    for row in range(0,H):
        for col in range(0,W):
            for feature in range(0,L):
                counts = np.zeros(T, dtype = np.uint32)
                for n in neighbors[row][col]: # list of indices of the neighbours
                    if ((n[0] >= 0) and (n[0] < H) and (n[1] >= 0) and (n[1] < H)):
                        counts[int(data[n[0]][n[1]][feature])] += 1 # the neighbour's array of features
                inds_max = np.where(counts == np.max(counts))[0]
                media_message[row][col][feature] = rng.choice(inds_max)

@njit
def updateMediaMessage_np():
    
    media_message = np.zeros((H,W,L), dtype = np.uint32)
         
    for row in range(0,H):
        for col in range(0,W):
            for feature in range(0,L):
                counts = np.zeros(T, dtype = np.uint32)
                for n in neighbors[row][col]: # list of indices of the neighbours
                    if ((n[0] >= 0) and (n[0] < H) and (n[1] >= 0) and (n[1] < H)): 
                        counts[data[n[0]][n[1]][feature]] += 1 # the neighbour's array of features
                inds_max = np.where(counts == np.max(counts))[0]
                media_message[row][col][feature] = inds_max[np.random.randint(0, len(inds_max))]
   
    return media_message 

@njit
            
   
H = W = 50
L = 5
T = 10
S = 4

def numba_updateMessage_np():
    updateMediaMessage_np()
    
def normal_updateMessage_np():
    updateMediaMessage(rng)    

data, neighbors = createRandom_np()  

t1 = timeit(numba_updateMessage_np, number = 100)
print(f'{t1:.2f} secs for numba update_message')

# rng 
t2 = timeit(normal_updateMessage_np, number = 100)
print(f'{t2:.2f} secs for normal update_message')

        
                
    
       