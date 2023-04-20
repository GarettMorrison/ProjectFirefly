import serial
import sys
import os
import time
import struct
import math as m
import cv2
import matplotlib.pyplot as plt
import pickle as pkl
import imageio
from random import randint
import zmq
import sys
import numpy as np

import py.positionFuncs as pf


# Load input data
fileIn = open("data/posHistTEMP_DELETE_LATER.csv", "r")
positionData = []
metaData = []
for fooLine in fileIn.readlines():
    splt = fooLine.split(',')
    positionData.append([np.double(foo) for foo in splt[:3]])
    metaData.append([np.int32(foo) for foo in splt[3:-1]])

positionData = np.array(positionData)
metaData = np.array(metaData)


# Combine input data
rotationDur = []
driveDur = []
distance = []
angle = []

relativeX = []
relativeY = []

for ii in range(len(positionData)-1):
    # print(f"\n")

    # print(f"positionData:{positionData[ii]}")
    # print(f"metaData:{metaData[ii]}")

    if metaData[ii][0] +1 == metaData[ii+1][0]:
        rotationDur.append(metaData[ii][3])
        driveDur.append(metaData[ii][4])
        
        distOut = m.sqrt(m.pow(positionData[ii+1][0]-positionData[ii][0], 2) + m.pow(positionData[ii+1][1]-positionData[ii][1], 2) )
        distance.append(distOut)

        angleOut = np.arctan2(positionData[ii+1][0]-positionData[ii][0], positionData[ii+1][1]-positionData[ii][1], ) - positionData[ii][2]
        
        if angleOut > m.pi: angleOut -= m.pi*2
        if angleOut < -m.pi: angleOut += m.pi*2

        angle.append( angleOut)
        
        relativeX.append(m.sin(angleOut)*distOut)
        relativeY.append(m.cos(angleOut)*distOut)
    else:
        print(f"{metaData[ii][0]} != {metaData[ii+1][0]}")


rotationDur = np.array(rotationDur)
driveDur = np.array(driveDur)
distance = np.array(distance)
angle = np.array(angle)
relativeX = np.array(relativeX)
relativeY = np.array(relativeY)








# import pandas as pd

# from sklearn.datasets import load_boston
# from sklearn.preprocessing import KBinsDiscretizer


# boston = load_boston()
# boston_df = pd.DataFrame(boston['data'] )
# boston_df.columns = boston['feature_names']
# boston_df['PRICE'] = boston['target']

# n_bins = 5

# qt = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

# scaled_feature_names = [f"BIN_{x}" for x in boston['feature_names']]
# boston_df[scaled_feature_names] = qt.fit_transform(boston_df[boston['feature_names']])
# boston_df[scaled_feature_names] = boston_df[scaled_feature_names].astype(int)











# # 2D Plot
# plt.scatter(relativeX, relativeY)
# plt.show()


print(f"rotationDur:{rotationDur}")
print(f"driveDur:{driveDur}")
print(f"distance:{distance}")
print(f"angle:{angle}")



# 3d Plot
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
# ax2 = fig.add_subplot(111, projection='3d')

ax1.set_xlabel("rotationDur (mS)")
ax1.set_ylabel("driveDur (mS)")


# ax1.scatter(rotationDur, driveDur, distance)
# ax1.set_zlabel("distance (mm)")


ax1.scatter(rotationDur, driveDur, (angle+np.pi)*40)
ax1.set_zlabel("angle (rad)")

plt.show()