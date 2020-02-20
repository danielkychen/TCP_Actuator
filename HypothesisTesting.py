# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:10:48 2019

@author: Daniel

Script for testing:
    
1. Reaction time of haptic skin @250mm vs 274.3ms, the mean
reaction time of tactile stimulus from Forster et al.

2. Comparing the mean completion times of haptic skin and vibration motor
"""

import pandas as pd
import numpy as np
from scipy.stats import bartlett
from scipy.stats import ttest_ind
from scipy.stats import sem
from scipy.stats import t
#from scipy.stats import trim_mean
from statistics import mean
from matplotlib import pyplot
import seaborn as sns
#import bootstrapped.bootstrap as bs
#import bootstrapped.stats_functions as bs_stats
import random

#Functions-------------------------------------------------------------------
def boot_paired(x, y, N):
    
    MeanDiff = np.zeros(shape=(N,1))
    diff = x - y
    
    #array of N resamples of difference of means
    MeanDiff = bootstrap(diff, N)
    
    bootstrapMean = mean(MeanDiff)
    print('BootstrapMean = ', bootstrapMean)
    MeanDiff = trim10Percent(MeanDiff)
    
    pyplot.hist(MeanDiff)
    pyplot.show()
    
    dist = bootstrapMean - 0
    
    if (bootstrapMean > 0):
        p = (sum(MeanDiff > bootstrapMean+dist)+sum(MeanDiff == bootstrapMean+dist)
                +sum(MeanDiff < 0)+sum(MeanDiff == 0)+1)/(0.8*N+1)
    else:
        p = (sum(MeanDiff < bootstrapMean+dist)+sum(MeanDiff == bootstrapMean+dist)
                +sum(MeanDiff > 0)+sum(MeanDiff == 0)+1)/(0.8*N+1)
    
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
    Mean = trim10Percent(Mean)
    
    bootstrapMean = mean(Mean)
    print('BootstrapMean = ', bootstrapMean)
    
    pyplot.hist(Mean)
    pyplot.show()
    
    dist = bootstrapMean - parameter
    
    if (bootstrapMean > parameter):
        p = (sum(Mean > bootstrapMean+dist)+sum(Mean == bootstrapMean+dist)
                +sum(Mean < parameter)+sum(Mean == parameter)+1)/(0.8*N+1)
    else:
        p = (sum(Mean < bootstrapMean+dist)+sum(Mean == bootstrapMean+dist)
                +sum(Mean > parameter)+sum(Mean == parameter)+1)/(0.8*N+1)
    
    return p;

def trim10Percent (data):
    #input a sorted array
    #remove top and bottom 10% of values
    #Calculate index of bottom 10% and top 10%
    bottomIndex = round(len(data)*0.1)
    topIndex = round(len(data)*0.9)
    
    tempArray = data[bottomIndex+1:topIndex+1]
    
    return tempArray;

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a, ddof=n-1)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
#------------------------------------------------------------------------------
random.seed(1)


#Testing hypothesis 1----------------------------------------------------------
#Pull data in from Feedback Reaction Time.csv
ReactionDF = pd.read_csv('C:\\Users\\danth\\Documents\\Post Doc\\JB - Fiber Actuator\\FeedbackReactionTime.csv')

Stretch250 = np.array(ReactionDF['Stretch250'])
Stretch250 = Stretch250 - 0.005
Vibration250 = np.array(ReactionDF['Vibration250'])
Vibration250_corrected = Vibration250 - 0.065
Vibration250_corrected = Vibration250_corrected[Vibration250_corrected < 1]

Stretch250_outliered = Stretch250[Stretch250 < 1]


#Perform one sample bootstrap hypothesis test
p1 = boot_onesample(Stretch250_outliered, 0.2743, 10000)

#boot_mean, boot_lower, boot_upper = mean_confidence_interval(boot_Stretch250Mean)
#mean +/- 2*SE = 321.7ms +/- 48.3

#p3 = boot_paired(Stretch250, Vibration250_corrected, 1000)


#Testing hypothesis 2----------------------------------------------------------
CompletionDF = pd.read_csv('C:\\Users\\danth\\Documents\\Post Doc\\JB - Fiber Actuator\\Haptic Experiment Data\\HapticData.csv')

Stretch250_Comp = np.array(CompletionDF['Stretch250'])
Vibration250Comp = np.array(CompletionDF['Vibration250'])

p3 = boot_paired(Stretch250_Comp, Vibration250Comp, 1000)
















