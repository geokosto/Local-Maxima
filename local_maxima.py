# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Input
path = ["Datasets\DATA.csv","Datasets\DATA1.csv","Datasets\DATA2.csv","Datasets\DATA3.csv"]

df = pd.read_csv(path[1]) # Select dataset
x = df['2theta'].values
y = df['kapoio_yliko'].values

#Filter data
from scipy.signal import savgol_filter
y_filtered = savgol_filter(y, 91, 4)

#Find local minima and maxima
gradients=np.diff(y_filtered)
gradients = savgol_filter(gradients, 91, 3) # Tune those with respect to size of data
def cmp(a, b):
    return bool(a > b) - bool(a < b) 
maxima_num=0
minima_num=0
max_locations=[] 
min_locations=[]
count=0
for i in gradients[:-1]:
    count+=1
    if ((cmp(i,0)>0) & (cmp(gradients[count],0)<0) & (i != gradients[count])):
        maxima_num+=1
        max_locations.append(count)     
    if ((cmp(i,0)<0) & (cmp(gradients[count],0)>0) & (i != gradients[count])):
        minima_num+=1
        min_locations.append(count)
#max_locations.append(len(gradients))# Last value as maxima
#turning_max_vals = {'maxima_number':maxima_num,'minima_number':minima_num,'maxima_locations':max_locations,'minima_locations':min_locations}  

#Sort local maxima and find the corespond x and y values
y_max_val = y[max_locations]
x_max_val = x[max_locations]
max_vals=np.array([x_max_val,y_max_val])
max_vals = max_vals[ :, max_vals[0].argsort()]# Sort 2D numpy array by 2nd row
real_maxima = []
real_max_x = []
real_max_y = []

#Plot
plt.figure()
for i in range(maxima_num):
    n = 40  ## Tune this to adjust size of searching area
    if i==(maxima_num-1):
        lsy = y[max_locations[i] -n : len(y)]
        lsx = x[max_locations[i] -n : len(y)]
    else:
        lsy = y[max_locations[i] -n : max_locations[i]+n]
        lsx = x[max_locations[i] -n : max_locations[i]+n]

    plt.plot(lsx,lsy-((np.max(y)-np.min(y))*0.1))
    real_max_vals=np.array([lsy,lsx])
    real_max_vals = real_max_vals[ :, real_max_vals[0].argsort()]# Sort 2D numpy array by 2nd row
    real_max_y.append(real_max_vals[0][-1])
    real_max_x.append(real_max_vals[1][-1])
    
    real_maxima.append(real_max_vals)
    
num_of_maxima = 5 ## Tune this to specify total number of maximas
reals=np.array([real_max_x,real_max_y])
reals = reals[ :, reals[1].argsort()]# Sort 2D numpy array by 2nd row
real_max_x = reals[0][::-1]
real_max_y = reals[1][::-1]
real_max_x = real_max_x[:num_of_maxima]
real_max_y = real_max_y[:num_of_maxima]

plt.grid()
plt.xlim((x[0],x[-1]))
plt.title('Searching Areas')
plt.show()

#Plot
plt.figure()
plt.scatter(x,y,s=2,label = 'Original data')
plt.plot(x,y_filtered,'k-',label = 'Filtered data')
plt.scatter(real_max_x,real_max_y,color='r',label = 'Local maxima')
plt.grid()
plt.legend()
plt.show()

