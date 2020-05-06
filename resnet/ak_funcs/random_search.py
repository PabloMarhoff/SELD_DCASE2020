#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:08:51 2018

@author: adamkujawski
"""

import numpy as np
#import matplotlib.pyplot as plt

#def return_optimizer_values():
    
def dropout_value(min_dropout=0, max_dropout=0.5, round_decimals=2):    
    return np.round(np.random.uniform(min_dropout,max_dropout),round_decimals)    

def adam_optimizer_parameters(epsilon = 1e-08):

    learning_rates = np.arange(0.00001, 0.1, 0.00001)
    index = np.random.geometric(0.0005)
    while index > learning_rates.size:
        index = np.random.geometric(0.0005)
    lr = np.round(learning_rates[index],6)

    beta1 = np.round(np.random.normal(0.9,0.1),3)
    beta2 = np.round(np.random.normal(0.999,0.1),3) 
    
    while beta1 >= 1 or beta2 >= 1:
        beta1 = np.round(np.random.normal(0.9,0.1),3)
        beta2 = np.round(np.random.normal(0.999,0.1),3) 
        
    return lr,beta1,beta2,epsilon
    
def hidden_layer(min_layers=0, max_layers=3, min_units=2, max_units=4096):
    num_layers = np.int(np.random.uniform(min_layers,max_layers+1))  
    num_units = np.int(np.random.uniform(0,max_units+1))         
    return num_layers, num_units

#beta1 = []
#beta2 = []
#for i in range(1000):
#    beta1.append(np.random.normal(0.9,0.1))
#    beta2.append(np.random.normal(0.999,0.01))
    
#learning_rates = np.arange(0.00001, 0.1, 0.00001)
#lr = []
#for i in range(1000):
#    index = np.random.geometric(0.0005)
#    while index > learning_rates.size:
#        index = np.random.geometric(0.0005)
#    lr.append(learning_rates[index])   
#    
#plt.figure()
#plt.hist(lr,bins=100)