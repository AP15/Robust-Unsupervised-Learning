#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:22:54 2020

@author: andreapaudice
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import utility
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

filename = 'NG20'

# Create the training data
newsgroups_train = fetch_20newsgroups(subset='train', 
                                      remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
X_raw = np.matrix(vectors.toarray())

y = newsgroups_train.target

pca = PCA(n_components=6000)
pca.fit(X_raw)
X = pca.transform(X_raw)

# Create the test data
newsgroups_test = fetch_20newsgroups(subset='test', 
                                      remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_test.data)
X_raw_test = np.matrix(vectors.toarray())

X_test = pca.transform(X_raw_test)
y_test = newsgroups_test.target

# Save the dataset
np.savez('Data/kmeans/' + filename + '_dataset', X, y, X_test, y_test)

data = np.load('Data/kmeans/' + filename + '_dataset.npz')
X, y, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

inliers_per_class = 500
outliers_per_class = 50

# Inliers
ng1 = X[np.where(y==0)]
ng2 = X[np.where(y==1)]
X_1 = ng1[np.random.choice(len(ng1), inliers_per_class)]
X_2 = ng2[np.random.choice(len(ng2), inliers_per_class)]
A = X_1
A = np.vstack([A, X_2])

# Create outliers
for i in range(18):
    ngi = X[np.where(y==i+2)]
    X_out = ngi[np.random.choice(len(ngi), outliers_per_class)]
    A = np.vstack([A, X_out])
    
X_train = shuffle(A)

print('Training set size: %d'%(len(X_train)))
np.savez('Data/kmeans/' + filename + '_data', X_train)

# Run the experiment
data = np.load('Data/kmeans/' + filename + '_data.npz')
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

ng1 = X_test[np.where(y==0)]
ng2 = X_test[np.where(y==1)]

A = ng1
A = np.vstack([A, ng2])
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