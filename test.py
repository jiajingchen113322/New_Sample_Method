import numpy as np
from collections import Counter
import math
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
np.set_printoptions(suppress=True)

def visulize_3d(data_path):
    data=np.load('./temp_data/s1.npy')
    xyz_coor=data[:,:3]
    color_info=data[:,3:6]/255
    label=data[:,-1]
   
    #create_point_cloud
    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(xyz_coor)
    point_cloud.colors=o3d.utility.Vector3dVector(color_info) 

    o3d.visualization.draw_geometries([point_cloud])

if __name__=='__main__':
    data_path='D:\Computer_vision\\3D_Dataset\Stanford_Large_Scale\component_data_maker\Standford_component_data\Area_3\storage_1\storage_1.txt'
    data_path=data_path.replace('\\','/')
    visulize_3d(data_path)