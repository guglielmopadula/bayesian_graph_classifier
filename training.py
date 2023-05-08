from data.dataset import DiscData
import torch
from pytorch_lightning import Trainer
import numpy as np
from sklearn.metrics import confusion_matrix
from models.model import Discriminator
from models.graph_model import Discriminator
pos_data=np.load("data/edge_pos_values.npy")
neg_data=np.load("data/edge_neg_values.npy")
graph_index=np.load("data/edge_graph.npy")
graph_index=graph_index.T
graph_index=torch.tensor(graph_index)

data=DiscData(600,1,torch.tensor(pos_data,dtype=torch.float32),torch.tensor(neg_data,dtype=torch.float32),len(pos_data[0]),False)
model=Discriminator(data.latent_dim,data.latent_dim,data.batch_size,0.95,graph_index)


trainer = Trainer(max_epochs=500)
trainer.fit(model, data)
y_fit=np.array(model.predict(data.data_train[:]).detach()).reshape(-1)
y_pred=np.array(model.predict(data.data_test[:]).detach()).reshape(-1)
y_train=np.array(data.data_train[:][1].detach()).reshape(-1)
y_test=np.array(data.data_test[:][1].detach()).reshape(-1)
print(y_train.shape)
print(y_test.shape)
print(y_pred.shape)
print(y_fit.shape)
print(np.sum(np.abs(y_pred-y_fit))/len(y_test))
print(np.sum(np.abs(y_test-y_pred))/len(y_test))
