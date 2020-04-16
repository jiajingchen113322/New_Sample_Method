import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import math

def visu_point(file_path):
    txt_file=np.loadtxt(file_path)
    xyz_coor=txt_file[:,:3]
    color_info=txt_file[:,3:6]/256
    
    # make points here
    points_info=o3d.geometry.PointCloud()
    points_info.points=o3d.utility.Vector3dVector(xyz_coor)
    points_info.colors=o3d.utility.Vector3dVector(color_info)
    
    dcp=o3d.geometry.voxel_down_sample(points_info,0.05)
    # o3d.geometry.estimate_normals(dcp,search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.1, max_nn=30))
    o3d.geometry.estimate_normals(
        dcp,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                          max_nn=30))
    
    
    points_dic={}
    points_dic['points']=np.array(dcp.points)
    points_dic['color']=np.array(dcp.colors)
    points_dic['norm']=np.array(dcp.normals)
    np.save('./data/A1_office4.npy',points_dic)
    
    o3d.visualization.draw_geometries([dcp])
    

def load_point(point_path):
    points_dic=np.load(point_path,allow_pickle=True).item()
   
   #load basic information
    points_coor=points_dic['points']
    points_color=points_dic['color']
    points_norm=points_dic['norm']
    
    return points_coor,points_color,points_norm

def get_direction_list(points_norm):
    direction_info={}
    direction_list=[]
    
    for direct in range(8):
        direction_info['a{}'.format(direct)]=[]
    
    for indice,norm_vect in enumerate(points_norm):
        x,y,z=norm_vect
        
        #area 0 and 1
        if 0<=(y/(x+1e-6))<=1:
            if ((x>=0) and (z>=0)) or ((x<=0) and (z<=0)):
                direction_info['a0'].append(indice)
                direction_list.append(0)
                
            else: 
                direction_info['a1'].append(indice)
                direction_list.append(1)
        
        #area 2 and 3
        elif (y/(x+1e-6))>1:
            if ((x>=0) and (z>=0)) or ((x<=0) and (z<=0)):
                direction_info['a2'].append(indice)
                direction_list.append(2)
            
            else:
                direction_info['a3'].append(indice)
                direction_list.append(3)
        
        #area 4 5
        elif (y/(x+1e-6))<-1:
            if ((x<=0) and (z>=0)) or ((x>=0) and (z<=0)):
                direction_info['a4'].append(indice)
                direction_list.append(4)
            
            else: 
                direction_info['a5'].append(indice)
                direction_list.append(5)
        
        #area 6,7
        elif -1<(y/(x+1e-6))<0:
            if ((x<=0) and (z>=0)) or ((x>=0) and (z<=0)):
                direction_info['a6'].append(indice)
                direction_list.append(6)
            
            else:
                direction_info['a7'].append(indice)
                direction_list.append(7)
    
    
    
    # lens_list=[]
    # for key in direction_info.keys():
    #     lens_list.append(len(direction_info[key]))
    #     print(len(direction_info[key]))
    
    # lens_list=np.array(lens_list)
    # plt.bar(range(len(lens_list)),lens_list)
    # plt.show()

    return np.array(direction_list)


def cal_entropy(neighbor_area):
    count=Counter(neighbor_area)
    num_ele=len(neighbor_area)
    
    entropy=0
    for key in count.keys():
        num_appear=count[key]
        propotional=num_appear/num_ele
        entropy=-(propotional)*math.log(propotional,2)
    
    return entropy


def Shannon_entropy_list(point_cor,point_direction):
    nbr=NearestNeighbors(n_neighbors=30,algorithm='ball_tree').fit(point_cor)
    distance,indice=nbr.kneighbors(point_cor)
    
    #create entropy_list
    entroy_list=[]
    num_point=point_cor.shape[0]
    for point_indice in range(num_point):
        this_point_nei=indice[point_indice]
        nei_area=point_direction[this_point_nei]
        entropy=cal_entropy(nei_area)
        entroy_list.append(entropy)
    
    entroy_list=np.array(entroy_list)
    
    return entroy_list
    




def analyse_point(point_path):
    p_loc,p_color,p_norm=load_point(point_path)
    direction_list=get_direction_list(p_norm)
    entroy_list=Shannon_entropy_list(p_loc,direction_list)
    his=np.histogram(entroy_list,bins=5)
    
    
    #change_color
    change_loc=np.where(entroy_list<=0.1)[0]
    p_color[change_loc]=np.array([1,0,0])
    
    pointcloud=o3d.geometry.PointCloud()
    pointcloud.points=o3d.utility.Vector3dVector(p_loc)
    pointcloud.colors=o3d.utility.Vector3dVector(p_color)
    
    o3d.visualization.draw_geometries([pointcloud])

        


if __name__=='__main__':
    data_path='D:\Computer_vision/3D_Dataset\Stanford_Large_Scale\component_data_maker\Standford_component_data\Area_1\office_4/office_4.txt'
    data_path=data_path.replace('\\','/')
    # visu_point(data_path)
    analyse_point('./data/A1_office4.npy')