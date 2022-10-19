# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:19:12 2022

@author: nkyos
"""

from SIModel_massMedia_numpy_numba import Grid
import sys
import numpy as np
import os


# listLength = int(sys.argv[1])
# traits = int(sys.argv[2])
# dim = int(sys.argv[3])

# plots = True if int(sys.argv[4]) == 1 else False

# thresh_success_possMove = int(sys.argv[5])
# print_success_every = int(sys.argv[6])
# intermediateSaves = True if int(sys.argv[7]) == 1 else False

# media = int(sys.argv[8])
# media_message = None if len(sys.argv[9]) == 1 else np.array([int(i) for i in sys.argv[9]], dtype=np.uint32)
# B = float(sys.argv[10])

# n = int(sys.argv[11])



listLength = 5 #10
traits = 10 # below q_C for F=10; 30 is above q_C for F=10
dim = 50 #50 #30
plots = True

thresh_success_possMove = 100
print_success_every = 100
intermediateSaves = False
# seed = 12345

media = 2
media_message = None
B = 0.1

n = 10

grid = Grid(listLength, traits, dim,
            media, media_message, B
            )

# create folder with parameeters of run where to save results
cd = os.getcwd()
print(cd)
folder = grid.name
folder_path = os.path.join(cd, folder)
try: 
    os.mkdir(folder_path) 
except OSError as error: 
    print(error)  

os.chdir(folder_path)
cd = os.getcwd()
print(cd)

# run
grid.loop(n, plots, thresh_success_possMove, print_success_every,
          intermediateSaves)




