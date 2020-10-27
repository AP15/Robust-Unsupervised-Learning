# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:43:21 2020

@author: apaudice
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import perturb_images as PI
from sklearn.cluster import KMeans
import math
import RKM
import RPSA
import sphericalDepth as sd

def computeKMeansObj(X, centers):
    losses = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        temp = np.sum((centers - X[i, :])**2, axis=1)    
        losses[i] = np.min(temp) 
    return np.mean(losses)

def importFMNIST(data):
    if (data=='train'):
        df = pd.read_csv('Data/fMNIST/fashion-mnist_train.csv', sep=',')
    else:
        df = pd.read_csv('Data/fMNIST/fashion-mnist_test.csv', sep=',')
    
    # Normalize
    data = df.iloc[:, 1:785].values
    labels = df.iloc[:, 0].values
    imgs = np.asfarray(data[:, :]) / 255
    
    return imgs, labels

def importMNIST(data):
    if (data=='train'):
        df = pd.read_csv('MNIST/train.csv')
    else:
        df = pd.read_csv('MNIST/test.csv')

    # Normalize
    data = df.iloc[:, 1:785].values
    labels = df.iloc[:, 0].values
    imgs = np.asfarray(data[:, :]) / 255
    
    return imgs, labels

def importEMNIST(data):
    data = loadmat('EMNIST/emnist-byclass.mat')
    dataset = data['dataset']
    if (data == 'train'):
        a = dataset[0][0][0]
    else:
        a = dataset[0][0][1]           
    X = a[0][0][0]    
    b = a[0][0][1]
    y = np.zeros(len(b))
    
    for i in range(len(b)):
        y[i] = b[i][0] 
        
    rotation_angle = 90
    X_r = PI.rotatedigits(X, rotation_angle) / 255
    return X_r, y

def importFromFile(filename):
    gold = np.loadtxt(filename)
    d = int(gold.shape[1])
    X = gold[:, 0:d-1]
    y = gold[:, d-1].astype(int)
    return X, y

def plotObj(T, L, filename = 'dummy'):
    t = np.arange(0, T+1)
    f = plt.figure()
    plt.plot(t, L)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$L_t$')
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_obj.pdf", bbox_inches='tight')  
    
def km_var_z_exp(X_train):
    # Initialize algorithms
    T =  50
    it = 3
    tol = 1e-7
    k = 2
    M = int(len(X_train)/10)
    
    zeta_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tensor_rkm_centers = np.zeros((k, X_train.shape[1], len(zeta_range)))
    tensor_sd_centers = np.zeros((k, X_train.shape[1], len(zeta_range)))
    
    for i in range(len(zeta_range)):
        zeta = zeta_range[i]
        gamma = zeta_range[i]
        print('Zeta = %f'%(zeta))
        # Run RKM
        rkm = RKM.RKM(zeta, gamma, it, T, tol, k)
        rkm.fit(X_train)
        tensor_rkm_centers[:, :, i] = rkm.centers_
    
    # Run kmeans++
    kmpp = KMeans(n_clusters=k, init='k-means++', n_init=it, max_iter=T, 
                  tol=1e-04)
    kmpp.fit(X_train)
    kmpp_centers = kmpp.cluster_centers_
    
    # Run SD
    cleaner = sd.sphericalDepth(1, 'random', M)
    cleaner.fit(X_train)
    
    pi_count = np.argsort(cleaner.count_)
    X_train = X_train[pi_count]
    for i in range(len(zeta_range)):
        n_out = len(X_train) - math.floor(zeta_range[i]*len(X_train))
        X_clean = X_train[n_out:len(X_train)]
        kmpp.fit(X_clean)
        tensor_sd_centers[:, :, i] = kmpp.cluster_centers_
    
    return tensor_rkm_centers, kmpp_centers, tensor_sd_centers

def psa_var_z_exp(X_train):
    # Initialize algorithms
    T =  50
    it = 3
    tol = 1e-7
    k = 2
    M = int(len(X_train)/10)
    
    zeta_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tensor_rpsa_components = np.zeros((k, X_train.shape[1], len(zeta_range)))
    tensor_sd_components = np.zeros((k, X_train.shape[1], len(zeta_range)))
    
    for i in range(len(zeta_range)):
        zeta = zeta_range[i]
        gamma = zeta_range[i]
        print('Zeta = %f'%(zeta))
        # Run RKM
        rpsa = RPSA.RPSA(zeta, gamma, it, T, tol, k)
        rpsa.fit(X_train)
        tensor_rpsa_components[:, :, i] = rpsa.components_
    
    # Run PSA
    psa_components = rpsa.PCA(X_train)
    
    # Run SD
    cleaner = sd.sphericalDepth(1, 'random', M)
    cleaner.fit(X_train)
    
    pi_count = np.argsort(cleaner.count_)
    X_train = X_train[pi_count]
    for i in range(len(zeta_range)):
        n_out = len(X_train) - math.floor(zeta_range[i]*len(X_train))
        X_clean = X_train[n_out:len(X_train)]
        tensor_sd_components[:, :, i] = rpsa.PCA(X_clean)
    
    return tensor_rpsa_components, psa_components, tensor_sd_components