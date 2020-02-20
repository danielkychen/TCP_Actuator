# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:23:49 2019

@author: Daniel
"""

#This Script is for analysing data from the Haptic Experiment
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


#Function Definitions----------------------------------------
def plotHistogram(data):
    pyplot.hist(data)
    pyplot.show()
    return;
    
def bootstrap(data, resample_num):
    data = data.values
    bootstrapMeans = np.empty(0)
    #first generate 10,000 new samples
    for num in range(resample_num):
        array = np.empty(0)
        for n in range(len(data)):
            array = np.append(array,data[random.randint(0,13)])
            sampleMean = mean(array)
            
        bootstrapMeans = np.append(bootstrapMeans, sampleMean)
    
    return bootstrapMeans;

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a, ddof=n-1)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def bootstrapMeans(data, resample_num):
    data = data.values
    meanArray = np.zeros(shape=(resample_num,6))
    #tempDF = pd.DataFrame(data=,columns=['Vibration150','Vibration250','Vibration350','Stretch150','Stretch250','Stretch350'])
    for num in range(resample_num):
        array = np.zeros(shape=(14,6))
        for n in range(len(data)):
            array[n,] = data[random.randint(0,13),1:]
        
        tempDF = pd.DataFrame(data = array)
        meanArray[num,] = [mean(tempDF.iloc[:,0]), mean(tempDF.iloc[:,1]), mean(tempDF.iloc[:,2]), mean(tempDF.iloc[:,3]), mean(tempDF.iloc[:,4]), mean(tempDF.iloc[:,5])]
    
    return meanArray;

def CalculateDeviceDiff(data):
    #data is meanArray
    #change to dataframe
    tempDF = pd.DataFrame(data = data, columns=['Vibration150','Vibration250','Vibration350','Stretch150','Stretch250','Stretch350'])
    
    diffArray = np.zeros(shape=(len(tempDF),3))
    
    for n in range(len(tempDF)):
        diffArray[n,0] = tempDF.ix[n,'Vibration150'] - tempDF.ix[n,'Stretch150']
        diffArray[n,1] = tempDF.ix[n,'Vibration250'] - tempDF.ix[n,'Stretch250']
        diffArray[n,2] = tempDF.ix[n,'Vibration350'] - tempDF.ix[n,'Stretch350']
    
    diffDF = pd.DataFrame(data = diffArray, columns=['150Diff','250Diff','350Diff'])
    
    return diffDF;

def CalculateDistanceDiff(data):
    #data is meanArray
    #change to dataframe
    tempDF = pd.DataFrame(data = data, columns=['Vibration150','Vibration250','Vibration350','Stretch150','Stretch250','Stretch350'])
    
    diffArray = np.zeros(shape=(len(tempDF),6))
    
    for n in range(len(tempDF)):
        diffArray[n,0] = tempDF.ix[n,'Vibration150'] - tempDF.ix[n,'Vibration250']
        diffArray[n,1] = tempDF.ix[n,'Vibration150'] - tempDF.ix[n,'Vibration350']
        diffArray[n,2] = tempDF.ix[n,'Vibration250'] - tempDF.ix[n,'Vibration350']
        diffArray[n,3] = tempDF.ix[n,'Stretch150'] - tempDF.ix[n,'Stretch250']
        diffArray[n,4] = tempDF.ix[n,'Stretch150'] - tempDF.ix[n,'Stretch350']
        diffArray[n,5] = tempDF.ix[n,'Stretch250'] - tempDF.ix[n,'Stretch350']
    
    diffDF = pd.DataFrame(data = diffArray, columns=['V150_250','V150_350','V250_350','S150_250','S150_350','S250_350'])
    
    return diffDF;
    
def trim10Percent (data):
    #input a sorted array
    #remove top and bottom 10% of values
    #Calculate index of bottom 10% and top 10%
    bottomIndex = round(len(data)*0.1)
    topIndex = round(len(data)*0.9)
    
    tempArray = data[bottomIndex+1:topIndex+1]
    
    return tempArray;

#---------------------------------------------------------------------------------------
random.seed(1)

#import participant data
df = pd.read_csv('C:\\Users\\danth\\Documents\\Post Doc\\JB - Fiber Actuator\\Haptic Experiment Data\\HapticData.csv')

#Bootstrap each sample n = 10000 and compare means
meanDiffDF_Comp = CalculateDeviceDiff(bootstrapMeans(df, 10000))

#trim 10% and calculate 95% confidence interval
#plotHistogram(meanDiffDF['150Diff'])
#plotHistogram(meanDiffDF['250Diff'])
#plotHistogram(meanDiffDF['350Diff'])

trim150Diff_Comp = meanDiffDF_Comp.iloc[:,0].values
trim150Diff_Comp.sort()

trim250Diff_Comp = meanDiffDF_Comp.iloc[:,1].values
trim250Diff_Comp.sort()

trim350Diff_Comp = meanDiffDF_Comp.iloc[:,2].values
trim350Diff_Comp.sort()

trim150Diff_Comp = trim10Percent(trim150Diff_Comp)
trim250Diff_Comp = trim10Percent(trim250Diff_Comp)
trim350Diff_Comp = trim10Percent(trim350Diff_Comp)

boot150DiffMeanComp, boot150Diff_botComp, boot150Diff_upComp = mean_confidence_interval(trim150Diff_Comp)
boot250DiffMeanComp, boot250Diff_botComp, boot250Diff_upComp = mean_confidence_interval(trim250Diff_Comp)
boot350DiffMeanComp, boot350Diff_botComp, boot350Diff_upComp = mean_confidence_interval(trim350Diff_Comp)

plotHistogram(trim150Diff_Comp)
plotHistogram(trim250Diff_Comp)
plotHistogram(trim350Diff_Comp)

#-----------Comparing distance within device-------------------------
meanDiffDF_Distance = CalculateDistanceDiff(bootstrapMeans(df, 10000))

trimV150_250 = meanDiffDF_Distance.iloc[:,0].values
trimV150_250.sort()

trimV150_350 = meanDiffDF_Distance.iloc[:,1].values
trimV150_350.sort()

trimV250_350 = meanDiffDF_Distance.iloc[:,2].values
trimV250_350.sort()

trimS150_250 = meanDiffDF_Distance.iloc[:,3].values
trimS150_250.sort()

trimS150_350 = meanDiffDF_Distance.iloc[:,4].values
trimS150_350.sort()

trimS250_350 = meanDiffDF_Distance.iloc[:,5].values
trimS250_350.sort()

trimV150_250 = trim10Percent(trimV150_250)
trimV150_350 = trim10Percent(trimV150_350)
trimV250_350 = trim10Percent(trimV250_350)
trimS150_250 = trim10Percent(trimS150_250)
trimS150_350 = trim10Percent(trimS150_350)
trimS250_350 = trim10Percent(trimS250_350)

#plotHistogram(trimV150_250)
#plotHistogram(trimV150_350)
#plotHistogram(trimV250_350)
#plotHistogram(trimS150_250)
#plotHistogram(trimS150_350)
#plotHistogram(trimS250_350)

trimV150_250MeanComp, trimV150_250_botComp, trimV150_250_upComp = mean_confidence_interval(trimV150_250)
trimV150_350MeanComp, trimV150_350_botComp, trimV150_350_upComp = mean_confidence_interval(trimV150_350)
trimV250_350MeanComp, trimV250_350_botComp, trimV250_350_upComp = mean_confidence_interval(trimV250_350)

trimS150_250MeanComp, trimS150_250_botComp, trimS150_250_upComp = mean_confidence_interval(trimS150_250)
trimS150_350MeanComp, trimS150_350_botComp, trimS150_350_upComp = mean_confidence_interval(trimS150_350)
trimS250_350MeanComp, trimS250_350_botComp, trimS250_350_upComp = mean_confidence_interval(trimS250_350)

#Plot boxplot of completion times
tempArray = np.empty([0,3])

for n in range(len(df.index)):
    tempArray = np.vstack((tempArray,[df.iloc[n,1],'Vibration',150]))
    
for n in range(len(df.index)):
    tempArray = np.vstack((tempArray,[df.iloc[n,2],'Vibration',250]))
    
for n in range(len(df.index)):
    tempArray = np.vstack((tempArray,[df.iloc[n,3],'Vibration',350]))  
    
for n in range(len(df.index)):
    tempArray = np.vstack((tempArray,[df.iloc[n,4],'Stretch',150]))
    
for n in range(len(df.index)):
    tempArray = np.vstack((tempArray,[df.iloc[n,5],'Stretch',250]))  
    
for n in range(len(df.index)):
    tempArray = np.vstack((tempArray,[df.iloc[n,6],'Stretch',350]))  
    
CompletionDF = pd.DataFrame(data=tempArray,columns=['Completion Time (s)', 'Device','Distance (mm)'])
CompletionDF['Completion Time (s)'] = CompletionDF['Completion Time (s)'].astype(float)
CompletionDF['Distance (mm)'] = CompletionDF['Distance (mm)'].astype(int)
CompletionDF['Device'] = CompletionDF['Device'].astype(str)

#Box plots of the confidence intervals of bootstrapped means
#Boxplot = sns.boxplot(x="Distance (mm)",y="Completion Time (s)",hue="Device",data=CompletionDF)
Boxplot = sns.boxplot(x="Device",y="Completion Time (s)",data=CompletionDF)
fig = Boxplot.get_figure()
fig.savefig("CompletionBoxPlot.png", dpi=1000)

import os
print(os.getcwd())












