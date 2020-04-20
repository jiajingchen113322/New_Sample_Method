import numpy as np
from collections import Counter
import math
from sklearn.neighbors import NearestNeighbors

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.amax(a,0))