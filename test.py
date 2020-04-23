import numpy as np
import torch
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# itera=range(5)
# tqiter=tqdm(itera,ncols=100,unit='batch',leave=False,desc='val')
# for i in tqiter:
#     tqiter.set_description("loss is %.3f" %(i+1))
#     time.sleep(1)

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

tensorboard=SummaryWriter(log_dir='./tensorboard_file/{}'.format(TIMESTAMP))
for i in range(100):
    num=torch.FloatTensor([i])
    num=num*5+100
    tensorboard.add_scalar('loss',num.item(),i)


