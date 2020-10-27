#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:51:50 2020

@author: andreapaudice
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import utility

filename = 'FMNIST50'

# Create the training data
# Import Fashion MNIST
X, y = utility.importFMNIST('train')

inliers_per_class = 1000
outliers_per_class = 250

# Inliers
sneakers = X[np.where(y==7)]
trousers = X[np.where(y==1)]
X_sneakers = sneakers[np.random.choice(sneakers.shape[0], inliers_per_class)]
X_trousers = trousers[np.random.choice(trousers.shape[0], inliers_per_class)]

# Create outliers
sandals = X[np.where(y==5)]
boots = X[np.where(y==9)]
tops = X[np.where(y==0)]
pullovers = X[np.where(y==2)]
dresses = X[np.where(y==3)]
coats = X[np.where(y==4)]
shirts = X[np.where(y==6)]
bag = X[np.where(y==8)]

A = tops[np.random.choice(tops.shape[0], outliers_per_class)]
A = np.vstack([A, sandals[np.random.choice(sandals.shape[0], outliers_per_class)]])
A = np.vstack([A, boots[np.random.choice(boots.shape[0], outliers_per_class)]])
A = np.vstack([A, pullovers[np.random.choice(pullovers.shape[0], outliers_per_class)]])
A = np.vstack([A, dresses[np.random.choice(dresses.shape[0], outliers_per_class)]])
A = np.vstack([A, coats[np.random.choice(coats.shape[0], outliers_per_class)]])
A = np.vstack([A, shirts[np.random.choice(shirts.shape[0], outliers_per_class)]])
A = np.vstack([A, bag[np.random.choice(bag.shape[0], outliers_per_class)]])

A = np.vstack([A, X_sneakers])
A = np.vstack([A, X_trousers])
X_train = shuffle(A)

print('Training set size: %d'%(len(X_train)))
np.savez('Data/kmeans/' + filename + '_data', X_train)

fig1, axes1 = plt.subplots(10, 10, figsize=(1.5*10,2*4))
axes1 = axes1.flatten()
for i in range(10*10):
      ax = axes1[i]
      ax.imshow(X_train[np.random.choice(X_train.shape[0], 1)].reshape(28, 28), 
                cmap='gray_r')
plt.suptitle('A sample of training data')
plt.tight_layout()
plt.show()  

# Run the experiment
data = np.load('Data/' + filename + '_data.npz')
X_train = data['arr_0']
tensor_rkm_centers, kmpp_centers, tensor_sd_centers = utility.km_var_z_exp(X_train)
np.savez('Results/kmeans/' + filename + '_centers', tensor_rkm_centers, kmpp_centers, 
          tensor_sd_centers)

# Test centers
# Create the test data
data = np.load('Results/kmeans/' + filename + '_centers.npz')
tensor_rkm_centers, kmpp_centers, tensor_sd_centers = data['arr_0'],\
                                                      data['arr_1'],\
                                                      data['arr_2']

#Import Fashion MNIST test
X, y = utility.importFMNIST('test')

# Merge shoes classes
sneakers = X[np.where(y==7)]
trousers = X[np.where(y==1)]

A = sneakers
A = np.vstack([A, trousers])
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