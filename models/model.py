from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.ld import Latent_Discriminator_base
import itertools
from models.losses.losses import L2_loss,CE_loss
import torch

class Discriminator(LightningModule):
    
    def __init__(self,data_shape,latent_dim,batch_size,drop_prob,hidden_dim: int= 500, ae_hyp=0.999,**kwargs):
        super().__init__()
        self.ae_hyp=ae_hyp
        self.drop_prob=drop_prob
        self.latent_dim=latent_dim
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.data_shape = data_shape
        self.discriminator=Latent_Discriminator_base(latent_dim=latent_dim,hidden_dim=hidden_dim ,drop_prob=drop_prob)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x,y=batch
        y=y.reshape(-1)
        opt= self.optimizers()
        prob=self.discriminator(x)
        prob=prob.reshape(-1)
        tot_loss=CE_loss(prob,y)
        self.manual_backward(tot_loss)
        self.clip_gradients(opt, gradient_clip_val=0.1)
        opt.step()        
        return tot_loss
            
    def test_step(self, batch, batch_idx):
        x,y=batch
        y=y.reshape(-1)
        prob=self.discriminator(x)
        prob=prob.reshape(-1)
        tot_loss=CE_loss(prob,y)
        return tot_loss

    def predict(self,batch):
        x,y=batch
        prob=self.discriminator(x)
        return torch.ones_like(prob)*(prob>0.5)+torch.zeros_like(prob)*(prob<0.5)

    def configure_optimizers(self):
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-3)
        return [optimizer_disc], []


