import torch
import numpy as np
import open3d as o3d
import torch.utils.data as data
import os


np.random.seed(0)
class S3DISDataset(data.Dataset):
    def __init__(self,root,split):
        if split=='train':
            self.area_list=['Area_1','Area_2','Area_3','Area_4','Area_6']
        else:
            self.area_list=['Area_5']
        
        self.root=root
        self.batch_list=self.create_batch_list()
        
        
    def create_batch_list(self):
        all_batch_list=[]
        for area in self.area_list:
            area_path=os.path.join(self.root,area)
            room_list=os.listdir(area_path)
            for room in room_list:
                batch_folder_path=os.path.join(area_path,room,'novel_sample_batch')
                batch_list=os.listdir(batch_folder_path)
                for batch in batch_list:
                    batch_path=os.path.join(batch_folder_path,batch)
                    
    




if __name__=='__main__':
    data_path='D:\Computer_vision\\3D_Dataset\Stanford_Large_Scale\component_data_maker\Standford_component_data'
    dataset=S3DISDataset(data_path,split='train')