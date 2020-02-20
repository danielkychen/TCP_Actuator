# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:14:37 2019

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
    m, se = np.mean(a), sem(a)
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
    
def trim10Percent (data):
    #input a sorted array
    #remove top and bottom 10% of values
    #Calculate index of bottom 10% and top 10%
    bottomIndex = round(len(data)*0.1)
    topIndex = round(len(data)*0.9)
    
    tempArray = data[bottomIndex+1:topIndex+1]
    
    return tempArray;

def welch_ttest(x, y): 
    ## Welch-Satterthwaite Degrees of Freedom ##
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
   
    t, p = ttest_ind(x, y, equal_var = False)
    
    print("\n",
          f"Welch's t-test= {t:.4f}", "\n",
          f"p-value = {p:.4f}", "\n",
          f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}")
    return;
#------------------------------------------------------------

#import participant data
df = pd.read_csv('C:\\Users\\danth\\Documents\\Post Doc\\JB - Fiber Actuator\\Haptic Experiment Data\\HapticData.csv')

#What questions are we interested in?
#Is the mean completion time different at different distances?
#Is the mean completion time different at using different devices?


#Bootstrap each sample n = 10,000 and compare means
Vibration150 = bootstrap(df.iloc[:,1],10000)
Vibration250 = bootstrap(df.iloc[:,2],10000)
Vibration350 = bootstrap(df.iloc[:,3],10000)

Stretch150 = bootstrap(df.iloc[:,4],10000)
Stretch250 = bootstrap(df.iloc[:,5],10000)
Stretch350 = bootstrap(df.iloc[:,6],10000)

#check normality and variance, whether or not we can use ANOVA
print('Checking Normality')
plotHistogram(Vibration150)
plotHistogram(Vibration250)
plotHistogram(Vibration350)

plotHistogram(Stretch150)
plotHistogram(Stretch250)
plotHistogram(Stretch350)
    

print('Bartlett Test')
bartlett(Vibration150, Vibration250, Vibration350, Stretch150, Stretch250, Stretch350)

#We satisfy normality but variances are nonhomogenous
#Proceed with using Welch's t test for different pairs

#Is there are a difference in the mean completion times between devices at the same distance?
welch_ttest(Vibration150, Stretch150)
welch_ttest(Vibration250, Stretch250)
welch_ttest(Vibration350, Stretch350)

#Is there a difference between the mean completion times between the 
#Stretch Skin at different distances?
welch_ttest(Stretch150, Stretch250)
welch_ttest(Stretch150, Stretch350)
welch_ttest(Stretch350, Stretch250)

welch_ttest(Vibration150, Vibration250)
welch_ttest(Vibration150, Vibration350)
welch_ttest(Vibration350, Vibration250)

#Construct new dataframe from bootstrapped means
#btDataFrame = pd.DataFrame(data=[],columns=['Bootstrapped Means', 'Device','Distance'])
tempArray = np.empty([0,3])

for n in range(len(Vibration150)):
    tempArray = np.vstack((tempArray,[Vibration150[n],'Vibration',150]))
    
for n in range(len(Vibration250)):
    tempArray = np.vstack((tempArray,[Vibration250[n],'Vibration',250]))
    
for n in range(len(Vibration350)):
    tempArray = np.vstack((tempArray,[Vibration350[n],'Vibration',350]))  
    
for n in range(len(Stretch150)):
    tempArray = np.vstack((tempArray,[Stretch150[n],'Stretch',150]))
    
for n in range(len(Stretch250)):
    tempArray = np.vstack((tempArray,[Stretch250[n],'Stretch',250]))  
    
for n in range(len(Stretch350)):
    tempArray = np.vstack((tempArray,[Stretch350[n],'Stretch',350]))  
    
btDataFrame = pd.DataFrame(data=tempArray,columns=['Bootstrapped Means', 'Device','Distance'])
btDataFrame['Bootstrapped Means'] = btDataFrame['Bootstrapped Means'].astype(float)
btDataFrame['Distance'] = btDataFrame['Distance'].astype(int)
btDataFrame['Device'] = btDataFrame['Device'].astype(str)

btDataFrame['Bootstrapped Means'] = btDataFrame['Bootstrapped Means']/100


#Box plots of the confidence intervals of bootstrapped means
Boxplot = sns.boxplot(x="Distance",y="Bootstrapped Means",hue="Device",data=btDataFrame)
fig = Boxplot.get_figure()
fig.savefig("Boxplot.png", dpi=1000)

#Results
#100% of participants were able to complete the task before the timeout period of 5 minutes
#Welch's t-tests were performed between each feedback type (vibration and skin stretch) at each distance
#The null hypothesis being that there were no differences between the means
#The means between Vibration150 and Stretch150 had a statistically significant difference
#(Welch's t-test, t(10068.84) = -293.03, p < 0.001)
#The means between Vibration250 and Stretch250 had a statistically significant difference
#(Welch's t-test, t(10306.05) = -283.03, p < 0.001)
#The means between Vibration150 and Stretch150 had a statistically significant difference
#(Welch's t-test, t(11798.28) = -299.65, p < 0.001)



#Reaction Analysis----------------------------------------------------------------------------------------




#Pull data in from Feedback Reaction Time.csv
ReactionDF = pd.read_csv('C:\\Users\\danth\\Documents\\Post Doc\\JB - Fiber Actuator\\FeedbackReactionTime.csv')

#Bootstrap each sample n = 10000 and compare means
meanDiffDF = CalculateDeviceDiff(bootstrapMeans(ReactionDF, 1000))

#trim 10% and calculate 95% confidence interval
#plotHistogram(meanDiffDF['150Diff'])
#plotHistogram(meanDiffDF['250Diff'])
#plotHistogram(meanDiffDF['350Diff'])
trim150Diff = meanDiffDF.iloc[:,0].values
trim150Diff.sort()

trim250Diff = meanDiffDF.iloc[:,1].values
trim250Diff.sort()

trim350Diff = meanDiffDF.iloc[:,2].values
trim350Diff.sort()

trim150Diff = trim10Percent(trim150Diff)
trim250Diff = trim10Percent(trim250Diff)
trim350Diff = trim10Percent(trim350Diff)

boot150DiffMean, boot150Diff_bot, boot150Diff_up = mean_confidence_interval(trim150Diff)
boot250DiffMean, boot250Diff_bot, boot250Diff_up = mean_confidence_interval(trim250Diff)
boot350DiffMean, boot350Diff_bot, boot350Diff_up = mean_confidence_interval(trim350Diff)

#Construct new dataframe from bootstrapped means
#btDataFrame = pd.DataFrame(data=[],columns=['Bootstrapped Means', 'Device','Distance'])
tempArray = np.empty([0,3])

for n in range(len(ReactionVibration150)):
    tempArray = np.vstack((tempArray,[ReactionVibration150[n],'Vibration',150]))
    
for n in range(len(ReactionVibration250)):
    tempArray = np.vstack((tempArray,[ReactionVibration250[n],'Vibration',250]))
    
for n in range(len(ReactionVibration350)):
    tempArray = np.vstack((tempArray,[ReactionVibration350[n],'Vibration',350]))  
    
for n in range(len(ReactionStretch150)):
    tempArray = np.vstack((tempArray,[ReactionStretch150[n],'Stretch',150]))
    
for n in range(len(ReactionStretch250)):
    tempArray = np.vstack((tempArray,[ReactionStretch250[n],'Stretch',250]))  
    
for n in range(len(ReactionStretch350)):
    tempArray = np.vstack((tempArray,[ReactionStretch350[n],'Stretch',350]))  
    
btReactionDataFrame = pd.DataFrame(data=tempArray,columns=['Bootstrapped Means', 'Device','Distance'])
btReactionDataFrame['Bootstrapped Means'] = btReactionDataFrame['Bootstrapped Means'].astype(float)
btReactionDataFrame['Distance'] = btReactionDataFrame['Distance'].astype(int)
btReactionDataFrame['Device'] = btReactionDataFrame['Device'].astype(str)

btReactionDataFrame['Bootstrapped Means'] = btReactionDataFrame['Bootstrapped Means']

#Box plots of the confidence intervals of bootstrapped means
Boxplot = sns.boxplot(x="Distance",y="Bootstrapped Means",hue="Device",data=btReactionDataFrame)
fig = Boxplot.get_figure()
fig.savefig("ReactionBoxPlot.png", dpi=1000)