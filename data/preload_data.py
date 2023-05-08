#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""

import numpy as np
import meshio
from tqdm import trange
def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points.astype(np.float32)
    return points


NUM_SAMPLES=600
points=getinfo("data/bunny_0.ply")
a=np.zeros((NUM_SAMPLES,points.reshape(-1).shape[0]))
for i in trange(NUM_SAMPLES):
    a[i]=getinfo("data/bunny_{}.ply".format(i)).reshape(-1)
np.save("data/data.npy",a)

edges=np.load("data/edge_indices.npy")
edges_values=np.zeros((NUM_SAMPLES,len(edges)))
for i in trange(NUM_SAMPLES):
    tmp=a[i].reshape(-1,3)
    for j in range(len(edges)):
        edge=edges[j]
        edges_values[i,j]=np.linalg.norm(tmp[edge[0]]-tmp[edge[1]])

np.save("data/edge_pos_values.npy",edges_values)