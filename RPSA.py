#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:16:12 2020

@author: andreapaudice
"""
import numpy as np

class RPSA(object):
    
    def __init__(self, zeta = 0.125, gamma = 0.75, n_init = 1, T = 10, 
                 tol = 1e-5, k = 1):     
        #Parameters
        self.zeta = zeta
        self.gamma = gamma   
        self.n_init = n_init
        self.T = T
        self.tol = tol
        self.k = k
        
        #Attributes
        self.components_ = np.empty(1)
        self.weights_ = np.empty(1)
        self.losses_ = np.empty(1)
        self.L_ = 0
    
    def fit(self, X):
        self.LloydPSA(X)
        best_components = self.components_
        best_losses = self.losses_
        best_weights = self.weights_
        best_L = self.L_[-1]
        for i in range(self.n_init-1):
            self.LloydPSA(X)
            if (self.L_[-1] < best_L):
                best_components = self.components_
                best_losses = self.losses_
                best_weights = self.weights_
                best_L = self.L_[-1]
        self.components_ = best_components
        self.losses_ = best_losses
        self.weights_ = best_weights
        self.L_ = best_L
    
    def LloydPSA(self, X):
        self.L_ = []
        
        #Initialize components
        self.components_ = self.project(np.random.rand(X.shape[1], self.k))
        self.computeLosses(X)
        self.getWeights()
        self.computeObj()      
        print('R[%d] = %f'%(0, self.L_[0]))
        
        for i in range(self.T):
            print('Iteration', i)
            self.updateComponents(X)
            self.computeLosses(X) 
            self.getWeights()
            self.computeObj()
            print('R[%d] = %f'%(i+1, self.L_[i+1]))
            if (self.L_[i] - self.L_[i+1] < self.tol):
                print('Local Minimum Reached in %d iterations.'%(i+1))
                break
    
    def updateComponents(self, X):
        self.components_ = self.SVD(np.multiply(X, 
                               np.sqrt(self.weights_[:, np.newaxis])))
    
    def PSA(self, X):
        U = self.SVD(X)
        return U
    
    # Utilities
    def SVD(self, X):
        n = X.shape[0]
        d = X.shape[1]
        cov_mat = np.zeros((d, d))
        for i in range(n):
            cov_mat += np.matmul(X[i].reshape(d, 1), X[i].reshape(1, d)) 
        cov_mat = 1/n * cov_mat
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) 
                       for i in range(len(eigen_vals))]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        U = (eigen_pairs[0][1][:, np.newaxis])
        for i in range(1, self.k):
            U = np.hstack((U, (eigen_pairs[i][1][:, np.newaxis])))
        U = U.reshape(d, self.k)
        return U
    
    def computeLosses(self, X):        
        self.losses_ = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            projection = np.matmul(self.components_.T, X[i, :])
            inv_projection = np.matmul(self.components_, projection)
            self.losses_[i] = np.linalg.norm(X[i, :] - inv_projection)**2
    
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
            return ((1/(self.zeta - self.gamma)) * (p - self.gamma))/I
        else:
            return 0.0
        
    def computeObj(self):
        n = len(self.losses_)
        L = self.weights_.dot(self.losses_)
        self.L_.append(L/n)
    
    def project(self, U):
        n = U.shape[1]
        N = np.zeros(U.shape)
        N[:, 0] = self.normalize(U[:, 0])
        
        for i in range(1, n):
            Ui = U[:, i]
            for j in range(0, i):
                Uj = U[:, j]
                t = Ui.dot(Uj)
                Ui = Ui - t * Uj
            N[:, i] = self.normalize(Ui)
        return N
    
    def normalize(self, v):
        return v / np.sqrt(v.dot(v))