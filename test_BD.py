#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:29:40 2020

@author: andreapaudice
"""

import numpy as np
import matplotlib.pyplot as plt                             
import ballDepth as BD

######################################################################
#RUN SCRIPT

k = 3
n = 300

#Data creation
mean = np.array([-1, 0])
cov = np.array([[0.1, 0], [0, 0.1]])
X_0 = np.random.multivariate_normal(mean, cov, 50)
mean = np.array([3, 0])
cov = np.array([[0.5, 0], [0, 0.5]])
X_1 = np.random.multivariate_normal(mean, cov, 50)
mean = np.array([-10, -10])
cov = np.array([[1, 0], [0, 1]])
X_out = np.random.multivariate_normal(mean, cov, 10)
A = X_0
A = np.vstack([A, X_1])
A = np.vstack([A, X_out])
X_train = A

z = 0.9
cleaner = BD.ballDepth(z)
X_clean = cleaner.fit(X_train)

z = 0.9
cleaner = BD.ballDepth(z, 'random', 75)
X_clean_r = cleaner.fit(X_train)

plt.figure()
plt.scatter(X_0[:,0], X_0[:,1], s=25, marker='o', c='blue')
plt.scatter(X_1[:,0], X_1[:,1], s=25, marker='o', c='blue')
plt.scatter(X_out[:,0], X_out[:,1], s=25, marker='o', c='red')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.figure()
plt.scatter(X_clean[:,0], X_clean[:,1], s=25, marker='o', c='blue')    
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.figure()
plt.scatter(X_clean_r[:,0], X_clean_r[:,1], s=25, marker='o', c='blue')    
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)