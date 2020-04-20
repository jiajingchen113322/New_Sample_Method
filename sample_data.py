import numpy as np
import os
import open3d as o3d
from utils.point_vis import create_new_input 
from utils import indoor_3d_util
import shutil


def create_sample_data(data_path,NUM_POINT):
    area_list=os.listdir(data_path)
    for area in area_list:
        area_path=os.path.join(data_path,area)
        room_list=os.listdir(area_path)
        for room in room_list:
            room_path=os.path.join(area_path,room)
            room_file_list=os.listdir(room_path)
            
            #create novel_batch_folder
            novel_batch_folder_path=os.path.join(room_path,'novel_sample_batch')
            if os.path.exists(novel_batch_folder_path):
                shutil.rmtree(novel_batch_folder_path)
                os.mkdir(novel_batch_folder_path)
            else:
                os.mkdir(novel_batch_folder_path)
            
            for fil in room_file_list:
                if fil[-3:]=='txt':
                    rooomtxt_file=os.path.join(room_path,fil)
                    #这里使用的是样本数据
                    # new_inpt=create_new_input(rooomtxt_file)
                    new_inpt=np.load('./temp_data/conf1.npy')
                    batch_data,label=indoor_3d_util.room2blocks_wrapper_normalized(new_inpt, NUM_POINT, block_size=1.0, stride=0.5, random_sample=False, sample_num=None)
                    s
    # new_inpt=create_new_input(data_path)
    # s
    

if __name__=='__main__':
    data_path='D:\Computer_vision\\3D_Dataset\Stanford_Large_Scale\component_data_maker\Standford_component_data'
    data_path=data_path.replace('\\','/')
    create_sample_data(data_path,4096)