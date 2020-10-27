# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:40:00 2020

@author: apaudice
"""

import numpy as np

class RKM(object):
    
    def __init__(self, zeta = 0.125, gamma = 0.75, n_init = 1, T = 5, 
                 tol = 1e-5, k = 1):        
        # Parameters
        self.zeta = zeta
        self.gamma = gamma   
        self.n_init = n_init
        self.T = T
        self.tol = tol
        self.k = k
        
        # Attributes
        self.centers_ = np.empty(1)
        self.clusters_ = np.empty(1)
        self.weights_ = np.empty(1)
        self.losses_ = np.empty(1)
        self.L_ = 0
    
    def fit(self, X):        
        self.LloydKMeans(X)
        best_centers = self.centers_
        best_clusters = self.clusters_
        best_losses = self.losses_
        best_weights = self.weights_
        best_L = self.L_[-1]
        for i in range(self.n_init-1):
            self.LloydKMeans(X)
            if (self.L_[-1] < best_L):
                best_centers = self.centers_
                best_clusters = self.clusters_
                best_losses = self.losses_
                best_weights = self.weights_
                best_L = self.L_[-1]
        self.centers_ = best_centers
        self.clusters_ = best_clusters
        self.losses_ = best_losses
        self.weights_ = best_weights
        self.L_ = best_L
    
    def LloydKMeans(self, X):
        self.L_ = []
        
        # Initialize centers                
        self.centers_ = X[np.random.choice(X.shape[0], self.k, False), :]
        self.find_clusters(X)
        self.getWeights()
        self.computeObj()
                
        for i in range(self.T):
            self.update_means(X)
            self.find_clusters(X)
            self.getWeights()
            self.computeObj()
            if (self.L_[i] - self.L_[i+1] < self.tol):
                print('Local Minimum Reached in %d iterations.'%(i+1))
                break
    
    def update_means(self, X):        
        for i in range(self.k):
            idxs_i = np.argwhere(self.clusters_==i)
            if (len(idxs_i) > 0):
                w_i = self.weights_[self.clusters_==i]
                X_i = np.diag(w_i).dot(X[self.clusters_==i, :])
                self.centers_[i, :] = 1/(np.sum(w_i))*np.sum(X_i, axis = 0)
    
    def find_clusters(self, X):
        n = X.shape[0]
        self.clusters_ = np.zeros(n)
        self.losses_ = np.zeros(n)
        
        for i in range(n):
            dist = np.sum((self.centers_ - X[i, :])**2, axis=1)
            self.clusters_[i] = np.argmin(dist)
            self.losses_[i] = np.min(dist)
    
    def getWeights(self):
        n = len(self.losses_)
        self.weights_ = np.zeros(n)
        pi = np.argsort(self.losses_)
        for i in range(n):
            self.weights_[pi[i]] = self.computeF(i/n)
                
    def computeF(self, p):
        I = 0.5 * (self.zeta + self.gamma)
        if (p <= self.zeta):
            return 1/I
        elif (p >= self.zeta and p <= self.gamma):
            return ((1/(self.zeta-self.gamma)) * (p - self.gamma))/I
        else:
            return 0.0
    
    def computeObj(self):
        n = len(self.losses_)
        L = self.weights_.dot(self.losses_)
        self.L_.append(L/n)