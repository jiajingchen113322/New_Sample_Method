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
                batch_folder_path=os.path.join(area_path,room,'Batch_Folder')
                batch_list=os.listdir(batch_folder_path)
                for batch in batch_list:
                    batch_path=os.path.join(batch_folder_path,batch)
                    all_batch_list.append(batch_path)
        
        return all_batch_list
    
    def __getitem__(self,batch_index):
        txt_file=self.batch_list[batch_index]
        data=np.loadtxt(txt_file)
        inpt=torch.FloatTensor(data[:,0:-1])
        label=torch.LongTensor(data[:,-1])
        return inpt,label

    def __len__(self):
        return len(self.batch_list)


def get_sets(data_path,batch_size):
    train_data=S3DISDataset(data_path,split='train')
    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=False,num_workers=2)

    test_data=S3DISDataset(data_path,split='test')
    test_loader=data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,num_workers=2)

    valid_loader=S3DISDataset(data_path,split='valid')
    valid_loader=data.DataLoader(dataset=valid_loader,batch_size=batch_size,shuffle=False,num_workers=2)
    
    return train_loader,test_loader,valid_loader



if __name__=='__main__':
    data_path='/data1/datasets/Standford_component_data'
    dataset=S3DISDataset(data_path,split='train')
    inpt,label=dataset[20]
    
