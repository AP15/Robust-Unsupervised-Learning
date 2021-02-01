#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:37:00 2020

@author: andreapaudice
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import utility

filename = 'iris'

df = pd.read_csv('https://archive.ics.uci.edu/ml/' \
                 'machine-learning-databases/iris/iris.data', header=None)
    
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

inliers_per_class = 30
outliers_per_class = 10

# Inliers
setosa = X[np.where( y=='Iris-setosa')]
#X_setosa = setosa[np.random.choice(setosa.shape[0], inliers_per_class)]
X_setosa = setosa[0:inliers_per_class]

# Create outliers
versicolor = X[np.where(y=='Iris-versicolor')]
virginica = X[np.where(y=='Iris-virginica')]
X_versicolor = versicolor[np.random.choice(versicolor.shape[0], outliers_per_class)]
X_virginica = virginica[np.random.choice(virginica.shape[0], outliers_per_class)]

A = X_setosa
A = np.vstack([A, X_versicolor])
A = np.vstack([A, X_virginica])
X_train = shuffle(A)

print('Training set size: %d'%(len(X_train)))
np.savez('Data/kmeans/' + filename + '_data', X_train)

# Run the experiment
k = 1
data = np.load('Data/kmeans/' + filename + '_data.npz')
X_train = data['arr_0']
tensor_rkm_centers, kmpp_centers, tensor_sd_centers = utility.km_var_z_exp(X_train, k)
np.savez('Results/kmeans/' + filename + '_centers', tensor_rkm_centers, kmpp_centers, 
          tensor_sd_centers)

# Test centers
# Create the test data
data = np.load('Results/kmeans/' + filename + '_centers.npz')
tensor_rkm_centers, kmpp_centers, tensor_sd_centers = data['arr_0'],\
                                                      data['arr_1'],\
                                                      data['arr_2']

#Create test data
X_test = setosa[inliers_per_class:]

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