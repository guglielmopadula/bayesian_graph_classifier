from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import ChebConv, TopKPooling
from models.basic_layers.ld import Latent_Discriminator_base
import itertools
from models.losses.losses import L2_loss,CE_loss
import torch

class Discriminator(LightningModule):
    
    def __init__(self,data_shape,latent_dim,batch_size,drop_prob,edge_index,hidden_dim: int= 500, ae_hyp=0.999,**kwargs):
        super().__init__()
        self.ae_hyp=ae_hyp
        self.drop_prob=drop_prob
        self.latent_dim=latent_dim
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.data_shape = data_shape
        self.edge_index=edge_index
        self.cheb1=ChebConv(-1,10,1)
        self.topk1=TopKPooling(100)
        self.cheb2=ChebConv(10,10,1)
        self.topk2=TopKPooling(50)
        self.cheb3=ChebConv(10,10,1)
        self.topk3=TopKPooling(10)
        self.cheb4=ChebConv(10,10,1)
        self.topk4=TopKPooling(5)
        self.cheb5=ChebConv(10,1,1)
        self.topk5=TopKPooling(3)
        self.flatten=nn.Flatten()
        self.sigmoid=nn.Sigmoid()
        self.lin=nn.Linear(64257,1)
        self.relu=nn.ReLU()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x,y=batch
        x=x.reshape(x.shape[0],-1,1)
        tmp=self.cheb1(x,self.edge_index)
        tmp=self.relu(tmp)
        graph=self.edge_index
        #tmp,graph=self.topk1(tmp,self.edge_index)
        tmp=self.cheb2(tmp,graph)
        tmp=self.relu(tmp)
        #tmp,graph=self.topk2(tmp,graph)
        tmp=self.cheb3(tmp,graph)
        tmp=self.relu(tmp)
        print(tmp.shape)
        #tmp,graph=self.topk3(tmp,graph)
        tmp=self.cheb4(tmp,graph)
        tmp=self.relu(tmp)
        #tmp,graph=self.topk4(tmp,graph)
        tmp=self.cheb5(tmp,graph)
        tmp=self.relu(x)
        #tmp,graph=self.topk5(tmp,graph)
        tmp=self.flatten(tmp)
        tmp=self.lin(tmp)
        prob=self.sigmoid(tmp)

        y=y.reshape(-1)
        opt= self.optimizers()
        prob=prob.reshape(-1)
        tot_loss=CE_loss(prob,y)
        self.manual_backward(tot_loss)
        self.clip_gradients(opt, gradient_clip_val=0.1)
        opt.step()        
        return tot_loss
            
    def test_step(self, batch, batch_idx):
        x,y=batch
        x,y=batch
        x=x.reshape(x.shape[0],-1,1)
        tmp=self.cheb1(x,self.edge_index)
        tmp=self.relu(tmp)
        graph=self.edge_index
        #tmp,graph=self.topk1(tmp,self.edge_index)
        tmp=self.cheb2(tmp,graph)
        tmp=self.relu(tmp)
        #tmp,graph=self.topk2(tmp,graph)
        tmp=self.cheb3(tmp,graph)
        tmp=self.relu(tmp)
        print(tmp.shape)
        #tmp,graph=self.topk3(tmp,graph)
        tmp=self.cheb4(tmp,graph)
        tmp=self.relu(tmp)
        #tmp,graph=self.topk4(tmp,graph)
        tmp=self.cheb5(tmp,graph)
        tmp=self.relu(x)
        #tmp,graph=self.topk5(tmp,graph)
        tmp=self.flatten(tmp)
        tmp=self.lin(tmp)
        prob=self.sigmoid(tmp)
        y=y.reshape(-1)
        prob=prob.reshape(-1)
        tot_loss=CE_loss(prob,y)
        return tot_loss

    def predict(self,batch):
        x,y=batch
        x=x.reshape(x.shape[0],-1,1)
        tmp=self.cheb1(x,self.edge_index)
        tmp=self.relu(tmp)
        graph=self.edge_index
        #tmp,graph=self.topk1(tmp,self.edge_index)
        tmp=self.cheb2(tmp,graph)
        tmp=self.relu(tmp)
        #tmp,graph=self.topk2(tmp,graph)
        tmp=self.cheb3(tmp,graph)
        tmp=self.relu(tmp)
        print(tmp.shape)
        #tmp,graph=self.topk3(tmp,graph)
        tmp=self.cheb4(tmp,graph)
        tmp=self.relu(tmp)
        #tmp,graph=self.topk4(tmp,graph)
        tmp=self.cheb5(tmp,graph)
        tmp=self.relu(x)
        #tmp,graph=self.topk5(tmp,graph)
        tmp=self.flatten(tmp)
        tmp=self.lin(tmp)
        prob=self.sigmoid(tmp)        
        return torch.ones_like(prob)*(prob>0.5)+torch.zeros_like(prob)*(prob<0.5)

    def configure_optimizers(self):
        optimizer_disc = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return [optimizer_disc], []


