import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import math
np.set_printoptions(suppress=True)


def get_point_info_dict(file_path):
    txt_file=np.loadtxt(file_path)
    xyz_coor=txt_file[:,:3]
    color_info=txt_file[:,3:6]
    label=txt_file[:,-1]
    
    # make points here
    points_info=o3d.geometry.PointCloud()
    points_info.points=o3d.utility.Vector3dVector(xyz_coor)
    points_info.colors=o3d.utility.Vector3dVector(color_info)
    
    # o3d.visualization.draw_geometries([points_info])
    
    dcp=o3d.geometry.voxel_down_sample(points_info,0.05)
    o3d.geometry.estimate_normals(
        dcp,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                          max_nn=30))
    
    # dcp_point=np.array(dcp.points)
    # nbrs=NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(xyz_coor)
    # dist,indice=nbrs.kneighbors(dcp_point)
    # indice=indice[:,1]
    # new_label=label[indice]    
    
    
    points_dic={}
    points_dic['points']=np.array(dcp.points)
    points_dic['color']=np.array(dcp.colors)
    points_dic['norm']=np.array(dcp.normals)
    
    return points_dic,txt_file
    # np.save('./data/A1_office_6.npy',points_dic)
    
    # o3d.visualization.draw_geometries([dcp])
    

def load_point(point_path):
    points_dic,original_points_array=get_point_info_dict(point_path)
    # points_dic=np.load(point_path,allow_pickle=True).item()
   
   #load basic information
    points_coor=points_dic['points']
    points_color=points_dic['color']
    points_norm=points_dic['norm']
    
    # o3d.visualization.draw_geometries([pointcloud])
    
    return points_coor,points_color,points_norm,original_points_array




def get_points_entropy(point_coordi,point_norm):
    #fistly we get cos similiarity for every point with its neighbor
    nbr=NearestNeighbors(n_neighbors=10,algorithm='ball_tree').fit(point_coordi)
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
    #input is a txt file which contains point xyz_coordinate,color,label
    #output is a array whose size is (num_point,8), [x,y,z,r,g,b,plane_label]
    
    p_loc,p_color,p_norm,original_points=load_point(point_path)
    original_loc=original_points[:,:3]
    uniform_norm=uniform_norm_sign(p_norm,p_loc)
   

    #create plane label
    entroy_list=get_points_entropy(p_loc,uniform_norm)
    plane_label=np.zeros((p_loc.shape[0]))
    plane_loc=np.where(entroy_list<=0.1)[0]
    plane_label[plane_loc]=1
    
    # new_inpt=np.concatenate([p_loc,p_color,plane_label[:,np.newaxis],p_label[:,np.newaxis]],axis=1)
    
    #Based on downsampling information, retrieve orginal dense points information
    nbrs=NearestNeighbors(n_neighbors=1,algorithm='ball_tree').fit(p_loc)
    dist,indic=nbrs.kneighbors(original_loc)
    original_plane_label=plane_label[indic.reshape(-1)]
    final_dense=np.concatenate([original_points,np.zeros_like(indic)],axis=1)
    final_dense[:,-1]=final_dense[:,-2]
    
    #final_dense[:6] is xyz and color information, [6] is plane label,[7] is segmentation label
    final_dense[:,-2]=original_plane_label
    
    
    # change_color
    #visulization part
   
    # final_points=final_dense[:,:3]
    # final_points_color=final_dense[:,3:6]/255
    # plane_loc=np.where(final_dense[:,6]==1)[0]
    # final_points_color[plane_loc]=np.array([1,0,0])

    # pointcloud=o3d.geometry.PointCloud()
    # pointcloud.points=o3d.utility.Vector3dVector(final_points)
    # pointcloud.colors=o3d.utility.Vector3dVector(final_points_color)
    # o3d.visualization.draw_geometries([pointcloud])
    
    #final_dense[:6] is xyz and color information, [6] is plane label,[7] is segmentation label
    return final_dense
        


if __name__=='__main__':
    data_path='D:\Computer_vision\\3D_Dataset\Stanford_Large_Scale\component_data_maker\Standford_component_data\Area_3\storage_1\storage_1.txt'
    data_path=data_path.replace('\\','/')
    # get_point_info_dict(data_path)
    # create_new_input('./data/A1_office_6.npy')
    dense_conf=create_new_input(data_path)
    np.save('./temp_data/dens_stor',dense_conf)