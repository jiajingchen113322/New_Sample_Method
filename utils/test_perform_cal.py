import numpy as np
import torch




def get_mean_accuracy(epsum,num_cls=13):
    labels=epsum['labels']
    labels=labels.reshape(-1)

    prediction=epsum['logits']
    prediction=prediction.reshape(-1,num_cls)
    prediction=np.argmax(prediction,1)
    
    similar=(labels==prediction).astype(int)
    mean_acc=np.sum(similar)/len(similar)
    
    return mean_acc





# if __name__=='__main__':
#     data=np.load('result.npy',allow_pickle=True).item()
#     get_mean_accuracy(data)

