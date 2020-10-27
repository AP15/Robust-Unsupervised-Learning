#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:42:23 2020

@author: andreapaudice
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import utility

filename = 'EMNIST50'

# Create the training data
# Import data
X, y, = utility.importEMNIST('train')
labels, count = np.unique(y, return_counts=True) 
X_0 = X[np.where(y==labels[0])]
X_1 = X[np.where(y==labels[1])]

# Create Dataset
inliers_per_class = 1000
outliers_per_class = 33

temp = X[np.where(y==labels[2])]
OUT = temp[np.random.choice(len(temp), outliers_per_class, False)]
for i in range(3, 62):
    temp = X[np.where(y==i)]
    OUT = np.vstack([OUT, temp[np.random.choice(len(temp), 
                                                 outliers_per_class, False)]])
    
A = X_0[np.random.choice(count[0], inliers_per_class)]
A = np.vstack([A, X_1[np.random.choice(count[1], inliers_per_class)]])
A = np.vstack([A, OUT])
X_train = shuffle(A) #Shuffle data

print('Dataset size: %d'%(len(X_train)))

np.savez('Data/kmeans/' + filename + '_data', X_train)

# Run the experiment
data = np.load('Data/' + filename + '_data.npz')
X_train = data['arr_0']
tensor_rkm_centers, kmpp_centers, tensor_sd_centers = utility.var_z_exp(X_train)
np.savez('Results/kmeans/' + filename + '_centers', tensor_rkm_centers, kmpp_centers, 
          tensor_sd_centers)

# Test centers
# Create the test data
data = np.load('Results/kmeans/' + filename + '_centers.npz')
tensor_rkm_centers, kmpp_centers, tensor_sd_centers = data['arr_0'],\
                                                      data['arr_1'],\
                                                      data['arr_2']

#Import EMNIST test
X, y = utility.importEMNIST('test')

# Merge shoes classes
zero = X[np.where(y==0)]
one = X[np.where(y==1)]

A = zero
A = np.vstack([A, one])
X_test = A

zeta_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rec_error_rkm = np.zeros(len(zeta_range))
rec_error_sd = np.zeros(len(zeta_range))

#Compute reconstructon error of RKM
for i in range(len(zeta_range)):
    rec_error_rkm[i] = utility.computeKMeansObj(X_test, 
                                                tensor_rkm_centers[:, :, i])
for i in range(len(zeta_range)):
    rec_error_sd[i] = utility.computeKMeansObj(X_test, 
                                                tensor_sd_centers[:, :, i])

#Compute reconstructon error of KM
rec_error_kmpp = utility.computeKMeansObj(X_test, kmpp_centers)
    
# Plot reconstruction error on test data
f = plt.figure()
plt.plot(zeta_range, rec_error_rkm)
plt.hlines(rec_error_kmpp, 0.1, 1.0, linestyle='--', linewidth=2,
            colors='blue', label='kmeans++')
plt.scatter(1, rec_error_kmpp, color='blue', marker='o')
plt.plot(zeta_range, rec_error_rkm, linewidth=2, 
          marker='o', color='red', label='RKM')
plt.plot(zeta_range, rec_error_sd, linestyle=':', linewidth=2,
          marker='o', color='green', label='SD')
plt.xlabel(r'$\zeta$', fontsize=12)
plt.ylabel(r'Test error', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()

# Show centers
fig, axarr = plt.subplots(2, 3, figsize=(15, 15))
ax = axarr[0, 0]
ax.imshow(tensor_rkm_centers[1, :, 4].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('RKM', fontsize=30)
ax = axarr[1, 0]
ax.imshow(tensor_rkm_centers[0, :, 4].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])

ax = axarr[0, 1]
ax.imshow(tensor_sd_centers[0, :, 4].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('SD', fontsize=30)
ax = axarr[1, 1]
ax.imshow(tensor_sd_centers[1, :, 4].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])

ax = axarr[0, 2]
ax.imshow(kmpp_centers[0, :].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('kmeans++', fontsize=30)
ax = axarr[1, 2]
ax.imshow(kmpp_centers[1, :].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])