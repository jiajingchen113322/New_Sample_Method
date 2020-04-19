import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import math
np.set_printoptions(suppress=True)


def visu_point(file_path):
    txt_file=np.loadtxt(file_path)
    xyz_coor=txt_file[:,:3]
    color_info=txt_file[:,3:6]/256
    label=txt_file[:,-1]
    
    # make points here
    points_info=o3d.geometry.PointCloud()
    points_info.points=o3d.utility.Vector3dVector(xyz_coor)
    points_info.colors=o3d.utility.Vector3dVector(color_info)
    
    o3d.visualization.draw_geometries([points_info])
    
    dcp=o3d.geometry.voxel_down_sample(points_info,0.05)
    o3d.geometry.estimate_normals(
        dcp,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                          max_nn=30))
    
    dcp_point=np.array(dcp.points)
    nbrs=NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(xyz_coor)
    dist,indice=nbrs.kneighbors(dcp_point)
    indice=indice[:,1]
    new_label=label[indice]    
    
    
    points_dic={}
    points_dic['points']=np.array(dcp.points)
    points_dic['color']=np.array(dcp.colors)
    points_dic['norm']=np.array(dcp.normals)
    points_dic['label']=new_label
    np.save('./data/A1_office_6.npy',points_dic)
    
    o3d.visualization.draw_geometries([dcp])
    

def load_point(point_path):
    points_dic=np.load(point_path,allow_pickle=True).item()
   
   #load basic information
    points_coor=points_dic['points']
    points_color=points_dic['color']
    points_norm=points_dic['norm']
    points_labe=points_dic['label']
    
    # o3d.visualization.draw_geometries([pointcloud])
    
    return points_coor,points_color,points_norm,points_labe




def get_points_entropy(point_coordi,point_norm):
    #fistly we get cos similiarity for every point with its neighbor
    nbr=NearestNeighbors(n_neighbors=30,algorithm='ball_tree').fit(point_coordi)
    distance,indice=nbr.kneighbors(point_coordi)
    num_point=point_coordi.shape[0]
    
    #get every point's entropy
    entropy_list=[]
    for point_indice in range(num_point):
        neighbor_indice=indice[point_indice]
        
        curent_norm=point_norm[point_indice]
        
        neigh_norm=point_norm[neighbor_indice]
        
        similarity=np.matmul(curent_norm,neigh_norm.T)
        similarity=np.clip(similarity,-1,1)
        degree_list=[]
        for i in range(len(similarity)):
            radio=math.acos(similarity[i])
            degree=math.degrees(radio)
            # if degree>90:
            #     degree=180-degree
            degree_list.append(degree)
        degree_list=np.array(degree_list)
        cu_entropy=single_point_entropy(degree_list)
        entropy_list.append(cu_entropy)
    
    entropy_list=np.array(entropy_list)
    return entropy_list
        
        
def single_point_entropy(degree_list):
    #devide 90 degree into 9 part, 10 degree each part
    part_list=(degree_list//10).astype(np.int32)
    count=Counter(part_list)
    num_ele=len(part_list)
    
    entropy=0
    for key in count.keys():
        num_appear=count[key]
        propotional=num_appear/num_ele
        entropy=-(propotional)*math.log(propotional,2)

    return np.abs(entropy)



def uniform_norm_sign(points_norm,points_coor):
    view_point=np.array([99,99,99]).reshape(-1,3)
    val=np.sum(points_norm*(view_point-points_coor),-1)
    signs=(val>0).astype(int)*2-1
    return -signs.reshape(-1, 1)*points_norm


def create_new_input(point_path):
    p_loc,p_color,p_norm,p_label=load_point(point_path)
    uniform_norm=uniform_norm_sign(p_norm,p_loc)
   

    #create plane label
    entroy_list=get_points_entropy(p_loc,uniform_norm)
    plane_label=np.zeros_like(p_label)
    plane_loc=np.where(entroy_list<=0.1)[0]
    plane_label[plane_loc]=1
    
    new_inpt=np.concatenate([p_loc,p_color,plane_label[:,np.newaxis],p_label[:,np.newaxis]],axis=1)
    return new_inpt
    
    
    #change_color
    # change_loc=np.where(entroy_list<=0.1)[0]
    # p_color[change_loc]=np.array([1,0,0])

    # pointcloud=o3d.geometry.PointCloud()
    # pointcloud.points=o3d.utility.Vector3dVector(p_loc)
    # pointcloud.colors=o3d.utility.Vector3dVector(p_color)
    # pointcloud.normals=o3d.utility.Vector3dVector(uniform_norm)
    
    # o3d.visualization.draw_geometries([pointcloud])

        


if __name__=='__main__':
    data_path='D:\Computer_vision/3D_Dataset\Stanford_Large_Scale\component_data_maker\Standford_component_data\Area_1\office_6/office_6.txt'
    data_path=data_path.replace('\\','/')
    # visu_point(data_path)
    create_new_input('./data/A1_office_6.npy')