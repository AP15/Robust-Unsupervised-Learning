#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:24:31 2020

@author: andreapaudice
"""

def kmeanspp(self, X):
    n = X.shape[0]
    d = X.shape[1]
    p = np.zeros(n)
    mu = np.zeros((self.k, d))
    
    # 2. Choose centers
    mu[0, :] = X[np.random.choice(n, 1), :]
    for i in range(self.k-1):
        for j in range(n):
            dist = np.min(np.sum((mu[0:i+1, :] - X[j, :])**2, axis=1))
            p[j] = dist
        prob = p/(np.sum(p))
        mu[i+1, :] = X[np.random.choice(n, 1, p = prob), :]
        #mu[i+1, :] = X[np.argmax(prob), :]
    return mu    