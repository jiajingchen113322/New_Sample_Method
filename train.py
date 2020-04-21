import numpy as np
import torch
from Seg_model.pointNet2.Pointnet2 import get_model
from utils.data_loader import get_sets
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def main(data_path,train=True):
    torch.backends.cudnn.enabled=False
    
    #set seed 
    torch.random.seed()
    model=get_model(13)
    
    train_loader,test_loader,valid_loader=get_sets(data_path,batch_size=16)
    
    if train:
        train_model(model,train_loader,valid_loader)
    
    if not train:
        # test(model,test_loader)
        pass


def train_model(model,train_loader,valid_loader):
    assert torch.cuda.is_available()
    
    #这里应该用GPU
    # device=torch.device('cuda:0')
    # model=model.to(device)
    
    device=torch.device('cpu')
    model=model.to(device)
    

    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)





    #here we define train_one_epoch
    def train_one_epoch():
        iterations=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        
        #真正训练这里应该解封
        # epsum=run_one_epoch(model,iterations,"train",loss_func=loss_func,optimizer=optimizer,loss_interval=10)
        
        epsum={'losses':2.0}
        summary={"loss/train":np.mean(epsum['losses'])}
        return summary



    #build tensorboard
    initial_epoch=0
    training_epoch=58
    tensorboard=SummaryWriter(log_dir='./tensorboard_file')
    tqdm_epoch=tqdm(range(initial_epoch,training_epoch),unit='epoch',ncols=100)
    
    for e in tqdm_epoch:
        train_summery=train_one_epoch()
        s



def run_one_epoch(model,tqdm_iter,mode,loss_func=None,optimizer=None,loss_interval=10):
    if mode=='train':
        model.train()
    else:
        model.eval()
        param_grads=[]
        for param in model.parameters():
            param_grads+=[param.requires_grad]
            param.requires_grad=False
    
    summary={"losses":[],"logits":[],"labels":[]}
    device=next(model.parameters()).device

    for i,(x_cpu,y_cpu) in enumerate(tqdm_iter):
        x,y=x_cpu.to(device),y_cpu.to(device)

        if mode=='train':
            optimizer.zero_grad()
        
        logits=model(x)[0]
        if loss_func is not None:
            loss=loss_func(logits.reshape(-1,logits.shape[-1]),y.view(-1))
            summary['losses']+=[loss.item()]
        
        if mode=='train':
            loss.backward()
            optimizer.step()

            #display
            if loss_func is not None and i%loss_interval==0:
                tqdm_iter.set_description("Loss: %.3f"%(np.mean(summary['losses'])))

        summary['logits']+=[logits.cpu().detach().numpy()]
        summary['labels']+=[y_cpu.numpy()]



    if mode!='train':
        for param,value in zip(model.parameters(),param_grads):
                param.requires_grad=value

    summary["logits"] = np.concatenate(summary["logits"], axis=0)
    summary["labels"] = np.concatenate(summary["labels"], axis=0)

    return summary


if __name__=='__main__':
    data_path='/data1/datasets/Standford_component_data'
    main(data_path)