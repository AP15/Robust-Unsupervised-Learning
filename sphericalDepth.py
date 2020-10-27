#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:54:31 2020

@author: andreapaudice
"""

import numpy as np
import itertools
import time

class sphericalDepth(object):
    
    def __init__(self, zeta = 0.125, mode = 'deterministic', M = 10):        
        # Parameters
        self.zeta = zeta
        self.mode = mode
        self.M = M
        
        # Attributes
        self.count_ = np.empty(1)
        
    def fit(self, X):        
        n = len(X)
        self.count_ = np.zeros(n)
        t_elapsed = 0
        t_last = 0
        
        idxs = set(range(n))
        subsets = np.array(list(itertools.combinations(idxs, 2)))
        for i in range(n):
            t0 = time.perf_counter()
            if(i%500 == 0):
                print('Point idx: %d'%(i))
                print("Time elapsed: %s"%(t_elapsed - t_last))
                t_last = t_elapsed
            if (self.mode == 'random'):
                idxs = np.random.choice(len(subsets), self.M)
                subsets_t = subsets[idxs]
            else:
                subsets_t = subsets
            for l in range(len(subsets_t)):
                #Check ball membership
                ss = list(subsets_t[l])
                d_ik = np.linalg.norm((X[i] - X[ss[0]])**2)
                d_il = np.linalg.norm((X[i] - X[ss[1]])**2)
                d_kl = np.linalg.norm((X[ss[0]] - X[ss[1]])**2)
                if (d_ik**2 + d_il**2 <= d_kl**2):
                    self.count_[i] += 1
            t1 = time.perf_counter() - t0
            t_elapsed += t1