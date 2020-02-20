# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 21:43:08 2019

@author: Daniel
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Script for plotting marker trajectory for figure in paper
#import csv
path = 'C:\\Users\\danth\\Documents\\Post Doc\\JB - Fiber Actuator\\PlotOfTrajectoryData.csv'
df = pd.read_csv(path,header=None,names=['Time (s)','Distance (mm)'])

Time = np.array(df.iloc[0:522,0].values)
Trajectory = np.array(df.iloc[277:799,1].values)
plt.xlabel('Time (s)')
plt.ylabel('Distance in the x-axis (mm)')
plt.plot(Time, Trajectory, color='b')
plt.axhline(y = 250, color='r', linestyle='dashed')
plt.axhline(y = 241.64, color='r', linestyle='dashed')
plt.axvline(x = 2.71, linestyle='dashed')
plt.axvline(x = 2.96, linestyle='dashed')
plt.axvline(x = 3.21, linestyle='dashed')
plt.axvline(x = 5.21, linestyle='dashed')

#First vertical line indicates when participant first passes target (250 mm)
#and feedback was on.
#Second vertical line indicates when the participant reverses direction and
#reacts to the feedback. Time difference between lines 1 - 2 = reaction time.
#Third vertical line indicates when feedback is turned off as participant reeneters
#behind the target.
#Fourth vertical line indicates the completion time. Time difference between lines 3 - 4
#is the 2 seconds the participant was required to keep their marker within
#the tolerance region before the trial would be deemed complete.
#Red horizontal lines indicate the tolerance region

#Save figure
#figure = plt.gcf()
#figure = plt.savefig('MarkerPlot.png',dpi=1000)

