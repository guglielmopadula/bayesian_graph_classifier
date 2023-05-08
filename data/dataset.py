import torch
import meshio
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import copy
from torch.utils.data import random_split



class DiscData(LightningDataModule):
    
    def get_size(self):
        return self.latent_dim

    def __init__(
        self,batch_size,num_workers,true_data,false_data,latent_dim,use_cuda):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.use_cuda=use_cuda
        self.num_workers = num_workers
        self.latent_dim=latent_dim
        n_true=len(true_data)
        n_false=len(false_data)
        y_true=torch.ones(n_true)
        y_false=torch.ones(n_false)
        x=torch.concatenate((true_data,false_data))
        y=torch.concatenate((y_true,y_false))
        self.data=torch.utils.data.TensorDataset(x,y)
        self.data_train,self.data_test=random_split(self.data,[len(self.data)//2,len(self.data)-len(self.data)//2])

    
    def prepare_data(self):
        pass


    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
          
  num_workers=self.num_workers)        
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

