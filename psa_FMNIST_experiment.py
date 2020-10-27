#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:17:54 2020

@author: andreapaudice
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import utility

filename = 'FMNIST50'

#Import Fashion MNIST
df_test = pd.read_csv('Data/fMNIST/fashion-mnist_train.csv', sep=',')

data_test = df_test.values
X = data_test[:, 1:]
y = data_test[:, 0]

inliers_per_class = 1000
outliers_per_class = 428

# Merge shoes classes
sandals = X[np.where(y==5)]
sneakers = X[np.where(y==7)]
boots = X[np.where(y==9)]
X_sandals = sandals[np.random.choice(sandals.shape[0], inliers_per_class)]
X_sneakers = sneakers[np.random.choice(sneakers.shape[0], inliers_per_class)]
X_boots = boots[np.random.choice(boots.shape[0], inliers_per_class)]

# Create outliers
tops = X[np.where(y==0)]
trousers = X[np.where(y==1)]
pullovers = X[np.where(y==2)]
dresses = X[np.where(y==3)]
coats = X[np.where(y==4)]
shirts = X[np.where(y==6)]
bag = X[np.where(y==8)]

A = tops[np.random.choice(tops.shape[0], outliers_per_class)]
A = np.vstack([A, trousers[np.random.choice(trousers.shape[0], outliers_per_class)]])
A = np.vstack([A, pullovers[np.random.choice(pullovers.shape[0], outliers_per_class)]])
A = np.vstack([A, dresses[np.random.choice(dresses.shape[0], outliers_per_class)]])
A = np.vstack([A, coats[np.random.choice(coats.shape[0], outliers_per_class)]])
A = np.vstack([A, shirts[np.random.choice(shirts.shape[0], outliers_per_class)]])
A = np.vstack([A, bag[np.random.choice(bag.shape[0], outliers_per_class)]])

A = np.vstack([A, X_sandals])
A = np.vstack([A, X_sneakers])
A = np.vstack([A, X_boots])
X_train = shuffle(A)

print('Training set size: %d'%(len(X_train)))
np.savez('Data/psa/' + filename + '_data', X_train)

fig1, axes1 = plt.subplots(3, 10, figsize=(1.5*10,2*4))
axes1 = axes1.flatten()
for i in range(3*10):
      ax = axes1[i]
      ax.imshow(X_train[np.random.choice(X_train.shape[0], 1)].reshape(28, 28), 
                cmap='gray_r')
plt.suptitle('A sample of training data')
plt.tight_layout()
plt.show()  


# Run the experiment
data = np.load('Data/psa/' + filename + '_data.npz')
X_train = data['arr_0']
tensor_rpsa_components, psa_components, tensor_sd_components = utility.psa_var_z_exp(X_train)
np.savez('Results/psa/' + filename + '_components', tensor_rpsa_components, 
         psa_components, tensor_sd_components)


# Test the components
X, y = utility.importFMNIST('test')
data = np.load('Results/psa/' + filename + '_components.npz')
tensor_rpsa_components, psa_components, tensor_sd_components = data['arr_0'], \
                                                               data['arr_1'], \
                                                               data['arr_2']

# Merge shoes classes
sandals = X[np.where(y==5)]
sneakers = X[np.where(y==7)]
boots = X[np.where(y==9)]

A = sandals
A = np.vstack([A, sneakers])
A = np.vstack([A, boots])
X_test = A

zeta_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rec_error_rpsa = np.zeros(len(zeta_range))
rec_error_sd = np.zeros(len(zeta_range))

#Compute reconstructon error of RKM
for i in range(len(zeta_range)):
    rec_error_rpsa[i] = utility.computePCAobj(X_test, 
                                                tensor_rpsa_components[:, :, i+4])
for i in range(len(zeta_range)):
    rec_error_sd[i] = utility.computePCAobj(X_test, 
                                                tensor_sd_components[:, :, i+4])

#Compute reconstructon error of KM
rec_error_psa = utility.computePCAobj(X_test, psa_components)
    
# Plot reconstruction error on test data
f = plt.figure()
plt.hlines(rec_error_psa, 0.5, 1.0, linestyle='--', linewidth=2,
            colors='blue', label='PSA')
plt.scatter(1, rec_error_psa, color='blue', marker='o')
plt.plot(zeta_range, rec_error_rpsa, linewidth=2, 
          marker='o', color='red', label='RPSA')
plt.plot(zeta_range, rec_error_sd, linestyle=':', linewidth=2,
          marker='o', color='green', label='SD')
plt.xlabel(r'$\zeta$', fontsize=12)
plt.ylabel(r'Test error', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()

# Show comps
fig, axarr = plt.subplots(2, 3, figsize=(15, 15))
ax = axarr[0, 0]
ax.imshow(tensor_rpsa_components[:, 0, 7].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('RPSA', fontsize=30)
ax = axarr[1, 0]
ax.imshow(tensor_rpsa_components[:, 1, 7].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])

ax = axarr[0, 1]
ax.imshow(tensor_sd_components[:, 0, 7].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('SD', fontsize=30)
ax = axarr[1, 1]
ax.imshow(tensor_sd_components[:, 1, 7].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])

ax = axarr[0, 2]
ax.imshow(psa_components[:, 0].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('PSA', fontsize=30)
ax = axarr[1, 2]
ax.imshow(psa_components[:, 1].reshape(28, 28), cmap='gray_r')
ax.set_xticks([])
ax.set_yticks([])