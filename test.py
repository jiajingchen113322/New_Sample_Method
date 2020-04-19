import numpy as np
from collections import Counter
import math
from sklearn.neighbors import NearestNeighbors

a=np.array([[0,0],[1,1],[2,2],[3,3]])
b=np.array([[-1,-1],[4,4]])

nbr=NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(a)
dis,indi=nbr.kneighbors(b)
print('s')