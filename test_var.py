#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:59:56 2020

@author: andreapaudice
"""

import numpy as np
import matplotlib.pyplot as plt
import RKM
from sklearn.cluster import KMeans                                       
import utility

######################################################################
#RUN SCRIPT

k = 3
n = 300

#Data creation
mean = np.array([-1, 0])
cov = np.array([[0.1, 0], [0, 0.1]])
X_0 = np.random.multivariate_normal(mean, cov, 50)
mean = np.array([3, 0])
cov = np.array([[0.5, 0], [0.5, 1]])
X_1 = np.random.multivariate_normal(mean, cov, 50)
mean = np.array([-1, -5])
cov = np.array([[5, 0], [0, 5]])
X_out = np.random.multivariate_normal(mean, cov, 50)
A = X_0
A = np.vstack([A, X_1])
A = np.vstack([A, X_out])
X_train = A

#Parameters for the emphasis function in Rlloyd
k=2
T=10
it=50
n_o=1

zeta_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
tensor_centers = np.zeros((k, X_train.shape[1], len(zeta_range)))

for i in range(len(zeta_range)):
    zeta = zeta_range[i]
    gamma = zeta
    print('Zeta = %f'%(zeta))
    # Run RKM
    rkm = RKM.RKM(zeta, gamma, it, T, k)
    tensor_centers[:, :, i], y_rkm, L = rkm.fit(X_train, n_o)

# Run kmeans++
kmpp = KMeans(n_clusters=k, init='k-means++', n_init=it, max_iter=T, 
              tol=1e-04)
kmpp.fit(X_train)
centers_km = kmpp.cluster_centers_

#Test set creation
mean = np.array([-1, 0])
cov = np.array([[0.1, 0], [0, 0.1]])
X_0 = np.random.multivariate_normal(mean, cov, 50)
mean = np.array([3, 0])
cov = np.array([[0.5, 0], [0, 0.5]])
X_1 = np.random.multivariate_normal(mean, cov, 50)
A = X_0
A = np.vstack([A, X_1])
X_test = A

rec_error_rkm = np.zeros(len(zeta_range))

#Compute reconstructon error of RKM
for i in range(len(zeta_range)):
    rec_error_rkm[i] = utility.computeKMeansObj(X_test, tensor_centers[:, :, i])

#Compute reconstructon error of KM
rec_error_km = utility.computeKMeansObj(X_test, centers_km)

# Plot reconstruction error on test data
f = plt.figure()
plt.plot(zeta_range, rec_error_rkm)
plt.hlines(rec_error_km, 0.1, 1.0, linestyle='--', 
            colors='blue', label='kmeans')
plt.scatter(1, rec_error_km, color='blue', marker='o')
plt.plot(zeta_range, rec_error_rkm, 
          marker='o', color='red', label='RKM')
plt.xlabel(r'$z$')
plt.ylabel(r'Test error')
plt.legend()
plt.grid()
plt.show()

# Show centers
for i in range(len(zeta_range)):
    plt.figure()
    plt.scatter(X_train[:,0], X_train[:,1], s=25, marker='o', c='blue')    
    plt.scatter(tensor_centers[:, 0, i], tensor_centers[:, 1, i], 
                marker='x', color = 'yellow', label='RKM', s=100)
    plt.scatter(centers_km[:,0], centers_km[:,1], marker='x', color = 'red', 
                label='KMmeans++', s=100)
    plt.legend(fontsize = 18)   
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(r'$z = %f$'%(zeta_range[i]))