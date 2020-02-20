# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:25:21 2019

@author: Daniel

This module contains functions for Bootstrap hypothesis testing based on Efron (1993)
"""

import numpy as np
import random
from matplotlib import pyplot
from statistics import mean

def boot_paired(x, y, N):
    
    MeanDiff = np.zeros(shape=(N,1))
    diff = x - y
    
    #array of N resamples of difference of means
    MeanDiff = bootstrap(diff, N)
    
    bootstrapMean = mean(MeanDiff)
    
    pyplot.hist(MeanDiff)
    pyplot.show()
    
    if (bootstrapMean > 0):
        p = (sum(MeanDiff > 2*bootstrapMean)+sum(MeanDiff == 2*bootstrapMean)
                +sum(MeanDiff < 0)+sum(MeanDiff == 0)+1)/(N+1)
    else:
        p = (sum(MeanDiff < 2*bootstrapMean)+sum(MeanDiff == 2*bootstrapMean)
                +sum(MeanDiff > 0)+sum(MeanDiff == 0)+1)/(N+1)
    
    return p;

def bootstrap(data, resample_num):
    #data = data.values
    bootstrapMeans = np.empty(0)
    #first generate 10,000 new samples
    for num in range(resample_num):
        array = np.empty(0)
        for n in range(len(data)):
            array = np.append(array,data[random.randint(0,len(data)-1)])
            sampleMean = mean(array)
            
        bootstrapMeans = np.append(bootstrapMeans, sampleMean)
    
    return bootstrapMeans;
    
def boot_onesample(x, parameter, N):
    
    Mean = np.zeros(shape=(N, 1))
    
    Mean = bootstrap(x, N)
    
    bootstrapMean = mean(Mean)
    
    pyplot.hist(Mean)
    pyplot.show()
    
    if (bootstrapMean > parameter):
        p = (sum(Mean > 2*bootstrapMean)+sum(Mean == 2*bootstrapMean)
                +sum(Mean < 0)+sum(Mean == 0)+1)/(N+1)
    else:
        p = (sum(Mean < 2*bootstrapMean)+sum(Mean == 2*bootstrapMean)
                +sum(Mean > 0)+sum(Mean == 0)+1)/(N+1)
    
    return p;
    
    
    
    
    
    
    