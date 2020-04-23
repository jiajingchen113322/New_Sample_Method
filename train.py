import numpy as np
import torch
from Seg_model.pointNet2.Pointnet2 import get_model


def main(train=True):
    torch.backends.cudnn.enabled=False
    
    #set seed 
    torch.random.seed(0)
    model=get_model()
    

if __name__=='__main__':
    main()