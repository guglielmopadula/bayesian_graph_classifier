#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 00:09:55 2023

@author: cyberguli
"""
import numpy as np
from tqdm import trange
pos_data=np.load("data/data.npy")
NUM_SAMPLES=len(pos_data)
pos_data=pos_data.reshape(NUM_SAMPLES,-1)
neg_data=pos_data.copy()
dim=pos_data.shape[1]
p=1/((np.arange(0,dim)+1)**0.5)
p=p/np.sum(p)

for i in trange(NUM_SAMPLES):
    index=np.random.choice(dim,1,p=p)+1
    indexes=np.random.choice(dim,index,replace=False)
    for j in indexes:
        neg_data[i,j]=neg_data[i,j]+(1+(1/index))*(-0.5+0.5*np.random.uniform())


np.save("data/edge_neg_values.npy",neg_data)

edges=np.load("data/edge_indices.npy")
edges_values=np.zeros((NUM_SAMPLES,len(edges)))
for i in range(NUM_SAMPLES):
    tmp=neg_data[i].reshape(-1,3)
    for j in range(len(edges)):
        edge=edges[j]
        edges_values[i,j]=np.linalg.norm(tmp[edge[0]]-tmp[edge[1]])


np.save("data/edge_neg_values.npy",edges_values)