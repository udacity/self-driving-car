# -------------------------------------------------------------------
# Challenge #2 - Data Exploration
# -------------------------------------------------------------------

# Creates plots of steering angles by consecutive timestamps
# By: cgundling 
# Rev Date: 11/19/16

from __future__ import print_function

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from pylab import *

def plotFeatures(data):
    # Plot all the time sections of steering data
    j = 0            # Rotating Color
    k = 0            # Rotating Marker
    jj = 0           # Subplot Number
    timebreak = [0]  # Store indices of timebreaks
    start = 0 
    c = ['r','b','g','k','m','y','c']
    marker = ['.','o','x','+','*','s','d']

    for i in range(1,data.shape[0]):
        if data[i,0] != data[i-1,0] and data[i,0] != (data[i-1,0] + 1):
            timebreak.append(int(data[i-1,0]))
            if jj < 70:
                jj = jj + 1
                print(jj)
                plt.subplot(7,10,jj)
                plt.plot(data[start:i-1,0],data[start:i-1,1],c=c[j],marker=marker[k])
                start = i
                j = j + 1
                if jj == 69:
                    plt.subplot(7,10,jj+1)
                    plt.plot(data[start:-1,0],data[start:-1,1],c=c[j],marker=marker[k])
            if j == 6:
                j = 0
                k = 0 #k = k + 1
            if k == 7:
                k = 0

    for i in range (1,71):
        plt.subplot(7,10,i)
        plt.xlabel('TimeStamp')
        plt.ylabel('Steering Angle')
        plt.grid(True)

    plt.suptitle('Consecutive Timestamp Steering')
    plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95,wspace=0.40,hspace=0.25)
    fig = plt.gcf()
    fig.set_size_inches(30, 15)
    fig.savefig('Steering.png', dpi=200)

# Main Program
def main():
    # Stats on steering data
    df_steer = pd.read_csv('dataset/steering.csv',usecols=['timestamp','angle'],index_col = False)
    u_A = str(len(list(set(df_steer['angle'].values.tolist()))))
    counts_A = df_steer['angle'].value_counts()

    # Mod the timestamp data
    time_factor = 10
    time_scale = int(1e9) / time_factor
    df_steer['time_mod'] = df_steer['timestamp'].astype(int) / time_scale
    u_T = str(len(list(set(df_steer['time_mod'].astype(int).values.tolist()))))

    # Some stats on steering angles/timestamps
    print('Number of unique steering angles...')
    print (u_A,df_steer.shape)
    print('Number of unique timestamps...')
    print (u_T,df_steer.shape)

    np.set_printoptions(suppress=False)
    counts_A.to_csv('counts.csv')
    df_steer['time_mod'].astype(int).to_csv('timestamps.csv',index=False)

    # Plot the steering data
    angle = np.zeros((df_steer.shape[0],1))
    time = np.zeros((df_steer.shape[0],1))

    angle[:,0] = df_steer['angle'].values
    time[:,0] = df_steer['time_mod'].values.astype(int)
    data = np.append(time,angle,axis=1)
    plotFeatures(data)

if __name__ == '__main__':
        main()
