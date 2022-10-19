import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from numba import njit
from scipy.ndimage import measurements as meas
    

@njit
def njit_getRandomItem(min_max):
    
    item = np.random.randint(min_max[0], min_max[1])
    
    return item

@njit
#Goes through the process of changing a neighbor
def njit_neighborChange_1(row, col, L, B, dim,
                        media_message, neighbors, data):
    '''
    for media = [0,1]
    
    '''

    #Media chosen with probability B else Random Neighbor Chosen
    if ((B > 0) and (np.random.random() < B)):
         data_neighbor = media_message.copy()
    else:
        neighborhood = njit_get_eff_neighborhood(dim, neighbors[row, col]) 
        neighbor = neighborhood[np.random.randint(0, neighborhood.shape[0]) ,:]
        
        data_neighbor = data[neighbor[0], neighbor[1]]  
    
    #Calculate probability of interaction
    overlap_bool = njit_overlap_neighbor(row, col, data_neighbor, data) # bool array
    overlap_size = np.sum(overlap_bool)      
    if ((overlap_size > 0) and (overlap_size < L)):
        
        #Interaction occurs is possible
        P2 = overlap_size / L # happens with this prob
        
        if np.random.random() < P2:
            
            #Choose random unshared feature between i and j
            diff_feats_inds = np.arange(L)[~overlap_bool]
            feature = np.random.choice(diff_feats_inds)
            data[row, col][feature] = data_neighbor[feature]
            
            return 1, data
    
    return 0, data


@njit
#Goes through the process of changing a neighbor
def njit_neighborChange_2(row, col, L, B, dim,
                        media_message, neighbors, data):
    '''
    for media = 2
    
    '''

    #Media chosen with probability B else Random Neighbor Chosen
    if ((B > 0) and (np.random.random() < B)):
         data_neighbor = media_message[row,col].copy()
    else:
        neighborhood = njit_get_eff_neighborhood(dim, neighbors[row, col]) 
        neighbor = neighborhood[np.random.randint(0, neighborhood.shape[0]) ,:]
        
        data_neighbor = data[neighbor[0], neighbor[1]]  
    
    #Calculate probability of interaction
    overlap_bool = njit_overlap_neighbor(row, col, data_neighbor, data) # bool array
    overlap_size = np.sum(overlap_bool)      
    if ((overlap_size > 0) and (overlap_size < L)):
        
        #Interaction occurs is possible
        P2 = overlap_size / L # happens with this prob
        
        if np.random.random() < P2:
            
            #Choose random unshared feature between i and j
            diff_feats_inds = np.arange(L)[~overlap_bool]
            feature = np.random.choice(diff_feats_inds)
            data[row, col][feature] = data_neighbor[feature]
            
            return 1, data
    
    return 0, data


@njit
def njit_overlap_neighbor(row, col, data_neighbor, data
                           ):
    
    return (data[row, col] == data_neighbor)

@njit
def njit_get_eff_neighborhood(H, neighborhood):
     '''
     filter out rows with nan values
     '''
     newn = np.zeros((neighborhood.shape[0],2), dtype = np.uint32)
     newn[:,:] = 1000
     for row in range(neighborhood.shape[0]):
         conc = True
         for col in range(neighborhood.shape[1]):
             if neighborhood[row, col] > H:
                 conc = False
         if conc:        
             if newn[0,0] == 1000:
                 newn = neighborhood[row, :].reshape(1,-1).copy()
             else:    
                 newn = np.concatenate((newn, neighborhood[row, :].reshape(1, -1)))
     
     return newn

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

@njit
def njit_createNeighborhood(row, col, H):
    '''
    out of range vals are 43435353535
    '''
    
    neighborhood = np.zeros((4,2), dtype = np.uint32)
    neighborhood[:,:] = -1
    
    if row+1 < H:
        neighborhood[0,:] = np.array([row+1,col], dtype = np.uint32)
    if row-1 > 0:
         neighborhood[1,:] = np.array([row-1,col], dtype = np.uint32)
    if col+1 < H:     
        neighborhood[2,:] = np.array([row,col+1], dtype = np.uint32)
    if col - 1 > 0:
        neighborhood[3,:] = np.array([row,col-1], dtype = np.uint32)   
        
    return neighborhood   


@njit
def njit_createRandom(H, W, T, L, neighbors,
                      data):

    for row in range(0,H):
        for col in range(0,W):
            neighbors[row,col] = njit_createNeighborhood(row, col, H)
            
            data[row][col] = np.random.randint(
                                low=0, high = T, size=L)
     
    return neighbors, data

@njit
def activeBond(data1, data2, L):
    
    overlap = 0
    for n in range(len(data1)):
        if data1[n] == data2[n]:
            overlap += 1
      
    if ((overlap > 0) and (overlap < L)):
    
        return True
    
    return False     
              
@njit(debug=True)       
def njit_updateMediaMessage_1(H, W, T, L, data, media_message, 
                       neighbors):
    '''
    for media type == 1
    '''
    
    for feature in range(0,L):
        counts = np.zeros(T, dtype = np.uint32)
        for row in range(0,H):
            for col in range(0,W): 
                counts[data[row][col][feature]] += 1 # all elements' array of features
        inds_max = np.where(counts == np.max(counts))[0].astype(np.uint32)
        media_message[feature] = inds_max[np.random.randint(
                            0, len(inds_max))]
        
    return media_message

@njit(debug=True)          
def njit_updateMediaMessage_2(H, W, T, L, data, media_message, 
                       neighbors):
    '''
    for media type == 2
    '''
        
    for row in range(0,H):
        for col in range(0,W):
            for feature in range(0,L):
                counts = np.zeros(T, dtype = np.uint32)
                neighborhood = neighbors[row][col]
                for n in neighborhood: # list of indices of the neighbours
                    if ((n[0] < H) and (n[1] < H)): # eff neighbs
                         counts[data[n[0]][n[1]][feature]] += 1 # the neighbour's array of features
                inds_max = np.where(counts == np.max(counts))[0].astype(np.uint32)
                media_message[row, col, feature] = inds_max[np.random.randint(
                                    0, len(inds_max))]
                   
    return media_message 
      
   

@njit
def njit_possibleMove_1(L, H, W, B, data, media_message,
                      neighbors):
    '''
    for media type in [0, 1] 
    
    TO-DO:
    Use this func to optimise the change neighbor: if it is not possible to 
    do a move in between neighbours, do not run prob to do media/neighbor
    but directly do media
    '''

    
    # compares agents to media
    if B > 0:
                      
        for row in range(0,H): 
            for col in range(0,W):
                    
                    bond = activeBond(data[row, col], media_message, L)    
                    if bond:
                        return True
                    
    if B < 1:
        
        # compares agents and their neighbors
        for row in range(0,H):
            for col in range(0,W):
                
                for n in neighbors[row,col]:
                    if ((n[0] < H) and (n[1] < H)): # eff neighbs
                        bond = activeBond(data[row, col], data[n[0],n[1]], L)  
                        if bond:
                            return True     
                        
    return False

@njit
def njit_possibleMove_2(L, H, W, B, data, media_message,
                      neighbors):
    '''
    
    for media type == 2
    
    TO-DO:
    Use this func to optimise the change neighbor: if it is not possible to 
    do a move in between neighbours, do not run prob to do media/neighbor
    but directly do media
    '''

    
    # compares agents to media
    if B > 0:
                      
        for row in range(0,H): 
            for col in range(0,W):
                
                bond = activeBond(data[row, col], media_message[row, col], L)    
                if bond:
                    return True 
                    
    if B < 1:
        
        # compares agents and their neighbors
        for row in range(0,H):
            for col in range(0,W):
                
                for n in neighbors[row,col]:
                    if ((n[0] < H) and (n[1] < H)): # eff neighbs
                        bond = activeBond(data[row, col], data[n[0],n[1]], L)  
                        if bond:
                            return True     
                        
    return False
     
          
             
                
#Grid Class

class Grid:

   #initializes Grid
   def __init__(self, listLength, traits, dim,
                media, media_message, B
                ):
       '''
       Parameters
       ----------
       media : int
           Media is one of 0, 1 or 2, standing for 0 = external, 1 = global, 2 = local
       media_message : list of lists - of 2D
           None if media in [1,2] bc it is endogenous
       B : float
           probability to interact with the media message, between 0 and 1
       random_seed : int
           seed to create the random generator
       '''
       assert media in [0,1,2]
       
       ## To-do:
           # - assert initial media message dimensions
        
       self.width = dim
       self.height = dim
       self.length = listLength # number of features
       self.traits = traits
       
       self.name = str(listLength) +'F'+ str(traits) +'Q'+ str(dim) + 'dim' \
           + str(media) + 'media' + str(media_message) + 'mess' + str(B) + 'B'
       
       self.data = np.zeros((dim,dim,listLength),dtype=np.uint32)
       self.neighbors = np.zeros((dim,dim,4,2),dtype=np.uint32)
       
       #
       self.media = media
       if media_message is None:
           self.media_message = np.zeros(listLength, dtype = np.uint32) \
                                   if media == 1 else \
                                   np.zeros((dim,dim,listLength), dtype = np.uint32)
       else:
           self.media_message = media_message
       self.media_message0 = media_message
                               
       self.B = B
       #

   #allows grid to be printed
   def __repr__(self):
       H = self.height
       W = self.width
       L = self.length
       s = ''   # the string to return
       for row in range(0,H):        
           for col in range(0,W):
               s += '|'
               for leng in range(0,L):
                   s+= str(self.data[row][col][leng])
           s += '\n'
       return s
                     
   def createRandom(self):
       
       H = self.height
       W = self.width
       T = self.traits
       L = self.length
       neighbors = self.neighbors
       data = self.data

       self.neighbors, self.data = njit_createRandom(H,W,T,L, neighbors, data)
        
       # define endogenous media message for initial configuration
       if self.media in [1, 2]: #global or local
           self.updateMediaMessage() 

   def get_eff_neighborhood(self, neighborhood):
        '''
        filter out rows with nan values
        '''
        H = self.height
        
        return neighborhood[(neighborhood < H).any(axis=1), :]        
                
   
   def updateMediaMessage(self):
        
         H = self.height
         W = self.width
         L = self.length
         T = self.traits
         media_message = self.media_message
         data = self.data
         neighbors = self.neighbors
         
         if self.media == 1:
             self.media_message = njit_updateMediaMessage_1(H, W, T, L, data, media_message, 
                                      neighbors) 
         else:
             self.media_message = njit_updateMediaMessage_2(H, W, T, L, data, media_message, 
                                      neighbors) 
         
      

   # #Returns a random integer representing a row
   # def getRandomRow(self):
       
   #     H = self.height
       
   #     row = np.random.randint(H)
       
   #     return row

   # #Returns a random interger representing a column
   # def getRandomCol(self):
       
   #     W = self.width
       
   #     col = np.random.randint(W)
       
   #     return col
   
   # Returns a boolean array of the features which are the same/not the same for both elements
   def overlap_neighbor(self, row, col, data_neighbor
                              ):
       
       overlap = np.equal(self.data[row][col], data_neighbor)
       
       return overlap


   #Returns true or false if there is a possible mov ein the grid  
   def possibleMove(self):

       L = self.length
       H = self.height
       W = self.width
       
       B = self.B
       data = self.data
       media_message = self.media_message
       neighbors = self.neighbors
       
       if self.media in [0, 1]:
           possMove = njit_possibleMove_1(L, H, W, B, data, media_message,
                                    neighbors)
       else:
           possMove = njit_possibleMove_2(L, H, W, B, data, media_message,
                                    neighbors)
       
       return possMove

   #Goes through the process of changing a neighbor
   def neighborChange(self, row, col):
       '''
       rands will be generated outside of this function
       bc I cant deal with extending the random generator
       
       '''

       L = self.length
       B = self.B
       dim = self.height
       media_message = self.media_message.copy()
       neighbors = self.neighbors.copy()
       data = self.data.copy()

       if self.media in [0,1]:
           change, new_data = njit_neighborChange_1(row, col, L, B, dim,
                               media_message, neighbors, data)
       else:
           change, new_data = njit_neighborChange_2(row, col, L, B, dim,
                               media_message, neighbors, data)
           
       self.data = new_data.copy()
       
       return change


   #Gathers statistics
   def loop(self, n, plots, thresh_success_possMove, print_success_every
            ):
       
       q=0
       results = np.empty((n,2)) # two cols: q and g
       
       #a loop of n
       while q!=n:
           
           self.createRandom()
           
           event=0
           success=0
           
           currMapName = self.name + str(q)+'Runs'+str(event)+'currMap'
           
           if plots == True:
               self.color(currMapName + '.png')
               plt.close()
           else:
               fpath = currMapName + '.pkl'
               with open(fpath, 'wb') as file:  
                   pkl.dump(self.data, file)
                   
           a=0
           b=0
           #a process
           while a==0:
               # row = self.getRandomRow()
               row = njit_getRandomItem((0, self.height))
               # col = self.getRandomCol()   
               col = njit_getRandomItem((0, self.height))
               
               b = self.neighborChange(row, col)
               success += b
               
               if (b == 1):
                   if (success%print_success_every == 0):
                       print(success)
                       
                       # if intermediateSaves:
                       #      currMapName = self.name + str(q)+'Runs'+str(event)+'currMap'
                               
                       #      if plots == True:
                       #          self.color(currMapName + '.png')
                       #          plt.close()
                       #      else:
                       #          fpath = currMapName + '.pkl'
                       #          with open(fpath, 'wb') as file:  
                       #              pkl.dump(self.data, file)
                           
                       
                   if ( (self.media != 0) and (self.B > 0) ):
                       self.updateMediaMessage()
                      
               event+=1 
               if success>=thresh_success_possMove and b!=0: # the numbers here is for the suggestive parameters set mentioned in the Introduction
                   possMove = self.possibleMove()
                   if possMove == False:
                       a=1       
           
           g, ng = self.analyzeClusters()
           results[q] = [self.traits, ng]
               
           finalMapName = self.name + str(q)+'Runs' + 'finalMap'
           if plots == True:
               self.color(finalMapName+'.png')
           else:
               # save final matrix
               fpath = finalMapName + '.pkl'
               with open(fpath, 'wb') as file:  
                  pkl.dump(self.data, file)

           plt.close('all')
           q+=1
           
       # write results to txt
       np.savetxt(self.name+'.txt', results, fmt='%-i')
           
   def analyzeClusters(self):
        '''
        returns the biggest cluster size (normalised) and the number of clusters
        '''

        dim = self.width
        # turn feature arrays to single values
        Matrix = njit_labelGrid(dim, self.data, self.length)
        
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
        N = dim**2
        norm_clusters_sizes = [i / N for i in clusters_sizes] # normalise sizes
        g = np.sort(norm_clusters_sizes)[::-1][0]   
        
        return g, num_clusters
                  

   #Returns a colored grid
   def color(self,imageName):
       
       dim = self.width
                  
       # turn feature arrays to single values
       Matrix = njit_labelGrid(dim, self.data, self.length)
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
       plt.savefig(imageName)
       plt.close()
    
        

