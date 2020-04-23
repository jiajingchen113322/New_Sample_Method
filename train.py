import numpy as np
import torch
from Seg_model.pointNet2.Pointnet2 import get_model
from utils.data_loader import get_sets
from utils.test_perform_cal import get_mean_accuracy
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())



def main(data_path,train=True):
    torch.backends.cudnn.enabled=False
    
    #set seed 
    # torch.random.seed()
    model=get_model(13,inpt_length=9)
    
    train_loader,test_loader,valid_loader=get_sets(data_path,batch_size=16)
    
    if train:
        train_model(model,train_loader,valid_loader)
    
    if not train:
        # test(model,test_loader)
        pass


def train_model(model,train_loader,valid_loader):
    assert torch.cuda.is_available()
    
    #这里应该用GPU
    device=torch.device('cuda:0')
    model=model.to(device)
    
    # device=torch.device('cpu')
    # model=model.to(device)
    

    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)





    #here we define train_one_epoch
    def train_one_epoch():
        iterations=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        
        #真正训练这里应该解封
        epsum=run_one_epoch(model,iterations,"train",loss_func=loss_func,optimizer=optimizer,loss_interval=10)
        
        summary={"loss/train":np.mean(epsum['losses'])}
        return summary


    def eval_one_epoch():
        iteration=tqdm(valid_loader,ncols=100,unit='batch',leave=False)
        #epsum only have logit and labes
        #epsum['logti'] is (batch,4096,13)
        #epsum['labels] is (batch,4096)
        
        epsum=run_one_epoch(model,iteration,"valid")
        mean_acc=get_mean_accuracy(epsum)
        summary={'meac':mean_acc}
        return summary



    #build tensorboard
    initial_epoch=0
    training_epoch=58
    tensorboard=SummaryWriter(log_dir='./tensorboard_file/test_on_{0}/{1}'.format(5,TIMESTAMP))
    tqdm_epoch=tqdm(range(initial_epoch,training_epoch),unit='epoch',ncols=100)
    
    for e in tqdm_epoch:
        train_summary=train_one_epoch()
        valid_summary=eval_one_epoch()
        summary={**train_summary,**valid_summary}

        #save checkpoint
        if (e%5==0) or (e==training_epoch-1):
            summary_saved={**train_summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict()}

            torch.save(summary_saved,'./pth_file/epoch_{}'.format(e))
        
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    


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