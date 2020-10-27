# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:03:22 2020

@author: apaudice
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def showsampledigits(digits):
    n = digits.shape[0]
                  
    # plot images
    num_row = 2
    num_col = 8
    fig1, axes1 = plt.subplots(num_row, num_col, 
                               figsize=(1.5*num_col,2*num_row))
    axes1 = axes1.flatten()
    for i in range(num_row*num_col):
          ax = axes1[i]
          ax.imshow(digits[np.random.choice(n, 1)].reshape(28, 28), cmap='gray_r')
    plt.suptitle('A sample of training data')
    plt.tight_layout()
    plt.show()

def zoomdigits(digits, zoom_magnitudes, show_image = False):
    n = digits.shape[0]
    d = digits.shape[1]
    X_R = np.zeros((n, d))
    datagen = ImageDataGenerator()
                  
    # Rotate the images
    for i in range(n):
        temp = digits[i].reshape(28, 28, 1)
        rotation = datagen.apply_transform(temp, transform_parameters = 
                                         {'zx':zoom_magnitudes[0],
                                          'zy':zoom_magnitudes[1]})
        X_R[i] = rotation[:, :, 0].reshape(1, d)
        
    if (show_image):
        # plot images
        num_row = 2
        num_col = 8
        fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
        for i in range(num_col):
              ax = axes1[0, i]
              ax.imshow(digits[i].reshape(28, 28), cmap='gray_r')
              ax = axes1[1, i]
              ax.imshow(X_R[i].reshape(28,28), cmap='gray_r')
              ax.set_title('Idx: {}'.format(i))
        plt.suptitle('Original and zoomed digits')
        plt.tight_layout()
        plt.show()
    
    return X_R

def sheardigits(digits, shear_angle, show_image = False):
    n = digits.shape[0]
    d = digits.shape[1]
    X_R = np.zeros((n, d))
    datagen = ImageDataGenerator()
                  
    # Rotate the images
    for i in range(n):
        temp = digits[i].reshape(28, 28, 1)
        rotation = datagen.apply_transform(temp, transform_parameters = 
                                         {'shear':shear_angle})
        X_R[i] = rotation[:, :, 0].reshape(1, d)
        
    if (show_image):
        # plot images
        num_row = 2
        num_col = 8
        fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
        for i in range(num_col):
              ax = axes1[0, i]
              ax.imshow(digits[i].reshape(28, 28), cmap='gray_r')
              ax = axes1[1, i]
              ax.imshow(X_R[i].reshape(28,28), cmap='gray_r')
              ax.set_title('Idx: {}'.format(i))
        plt.suptitle('Original and sheared digits')
        plt.tight_layout()
        plt.show()
    
    return X_R

def sheardigitsRandom(digits, shear_angle, show_image = False):
    n = digits.shape[0]
    d = digits.shape[1]
    X_R = np.zeros((n, d))
    datagen = ImageDataGenerator()
                  
    # Rotate the images
    for i in range(n):
        temp = digits[i].reshape(28, 28, 1)
        shear_angle = np.random.randint(45, 90)
        rotation = datagen.apply_transform(temp, transform_parameters = 
                                         {'shear':shear_angle})
        X_R[i] = rotation[:, :, 0].reshape(1, d)
        
    if (show_image):
        # plot images
        num_row = 2
        num_col = 8
        fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
        for i in range(num_col):
              ax = axes1[0, i]
              ax.imshow(digits[i].reshape(28, 28), cmap='gray_r')
              ax = axes1[1, i]
              ax.imshow(X_R[i].reshape(28,28), cmap='gray_r')
              ax.set_title('Idx: {}'.format(i))
        plt.suptitle('Original and sheared digits')
        plt.tight_layout()
        plt.show()
    
    return X_R


def rotatedigits(digits, rotation_angle, show_image = False):
    n = digits.shape[0]
    d = digits.shape[1]
    X_R = np.zeros((n, d))
    datagen = ImageDataGenerator()
                  
    # Rotate the images
    for i in range(n):
        temp = digits[i].reshape(28, 28, 1)
        rotation = datagen.apply_transform(temp, transform_parameters = 
                                         {'theta':rotation_angle})
        X_R[i] = rotation[:, :, 0].reshape(1, d)
        
    if (show_image):
        # plot images
        num_row = 2
        num_col = 8
        fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
        for i in range(num_col):
              ax = axes1[0, i]
              ax.imshow(digits[i].reshape(28, 28), cmap='gray_r')
              ax = axes1[1, i]
              ax.imshow(X_R[i].reshape(28,28), cmap='gray_r')
              ax.set_title('Idx: {}'.format(i))
        plt.suptitle('Original and rotated digits')
        plt.tight_layout()
        plt.show()
    
    return X_R

def rotatedigitsRandom(digits, rotation_angle, show_image = False):
    n = digits.shape[0]
    d = digits.shape[1]
    X_R = np.zeros((n, d))
    datagen = ImageDataGenerator()
                  
    # Rotate the images
    for i in range(n):
        temp = digits[i].reshape(28, 28, 1)
        shear_angle = np.random.randint(60, 90)
        rotation = datagen.apply_transform(temp, transform_parameters = 
                                         {'theta':shear_angle})
        X_R[i] = rotation[:, :, 0].reshape(1, d)
        
    if (show_image):
        # plot images
        num_row = 2
        num_col = 8
        fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
        for i in range(num_col):
              ax = axes1[0, i]
              ax.imshow(digits[i].reshape(28, 28), cmap='gray_r')
              ax = axes1[1, i]
              ax.imshow(X_R[i].reshape(28,28), cmap='gray_r')
              ax.set_title('Idx: {}'.format(i))
        plt.suptitle('Original and rotated digits')
        plt.tight_layout()
        plt.show()
    
    return X_R

def reflectdigits(digits, show_image = False):
    n = digits.shape[0]
    d = digits.shape[1]
    X_R = np.zeros((n, d))
    datagen = ImageDataGenerator()
                  
    # Rotate the images
    for i in range(n):
        temp = digits[i].reshape(28, 28, 1)
        reflection = datagen.apply_transform(temp, transform_parameters = 
                                         {'flip_horizontal':True})
        X_R[i] = reflection[:, :, 0].reshape(1, d)
        
    if (show_image):
        # plot images
        num_row = 2
        num_col = 8
        fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
        for i in range(num_col):
              ax = axes1[0, i]
              ax.imshow(digits[i].reshape(28, 28), cmap='gray_r')
              ax = axes1[1, i]
              ax.imshow(X_R[i].reshape(28,28), cmap='gray_r')
              ax.set_title('Idx: {}'.format(i))
        plt.suptitle('Original and rotated digits')
        plt.tight_layout()
        plt.show()
    
    return X_R