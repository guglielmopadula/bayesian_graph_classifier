#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 00:09:55 2023

@author: cyberguli
"""
from tqdm import trange
import numpy as np

triangles=np.load("tetras.npy")
def compute_edges_indices(graph):
    l=set([])
    for tetras in graph:
        for i in range(4):
            for j in range(i+1,4):
                l.add(tuple([min(tetras[i],tetras[j]),max(tetras[i],tetras[j])]))
    l=[list(s) for s in l]
    return l

edges=compute_edges_indices(triangles)

def compute_edge_graphs(edges):
    l=set([])
    for i in trange(len(edges)):
        for j in edges[i]:
            for k in range(len(edges)):
                if i!=k:
                    if j in edges[k]:
                        tmp=tuple([i,k])
                        l.add(tmp)
    l=[list(s) for s in l]
    return l

np.save("edge_indices.npy",edges)
edges_graph=compute_edge_graphs(edges)
np.save("edge_graph.npy",edges_graph)
