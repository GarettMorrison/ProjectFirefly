import os
import pickle as pkl
import numpy as np
import statistics as st
import sys

from py.positionFuncs import *

DO_PLOT_3D = False

if DO_PLOT_3D: import py.plot3D as plot3D

DO_PLOT_2D = not DO_PLOT_3D
if DO_PLOT_2D: 
    import matplotlib.pyplot as plt
    import random
    





dataPoints = {}

dataPoints['index'] = []
dataPoints['xPos'] = []
dataPoints['yPos'] = []
dataPoints['zPos'] = []
dataPoints['xRot'] = []
dataPoints['yRot'] = []
dataPoints['zRot'] = []
dataPoints['motionSel'] = []
dataPoints['motionDur'] = []



for fooLine in open("data/PositionLog.csv", "r").readlines():
    fooSplit = fooLine.split(',')

    dataPoints['index'].append(int(fooSplit[0]))
    dataPoints['xPos'].append(float(fooSplit[1]))
    dataPoints['yPos'].append(float(fooSplit[2]))
    dataPoints['zPos'].append(float(fooSplit[3]))
    dataPoints['xRot'].append(float(fooSplit[4]))
    dataPoints['yRot'].append(float(fooSplit[5]))
    dataPoints['zRot'].append(float(fooSplit[6]))
    dataPoints['motionSel'].append(fooSplit[7])
    dataPoints['motionDur'].append(int(fooSplit[8]))


dataPoints['xPos'] = np.array(dataPoints['xPos'])
dataPoints['yPos'] = np.array(dataPoints['yPos'])
dataPoints['zPos'] = np.array(dataPoints['zPos'])
dataPoints['xRot'] = np.array(dataPoints['xRot'])
dataPoints['yRot'] = np.array(dataPoints['yRot'])
dataPoints['zRot'] = np.array(dataPoints['zRot'])
dataPoints['motionSel'] = np.array(dataPoints['motionSel'])
dataPoints['motionDur'] = np.array(dataPoints['motionDur'])





motionPts = {}

for fooSel in "fblr":
    motionPts[fooSel] = {}
    motionPts[fooSel]['xPos'] = []
    motionPts[fooSel]['yPos'] = []
    motionPts[fooSel]['zPos'] = []
    motionPts[fooSel]['xRot'] = []
    motionPts[fooSel]['yRot'] = []
    motionPts[fooSel]['zRot'] = []
    motionPts[fooSel]['motionDur'] = []

print(motionPts)

# Save motions to CSV
moveCSV = open('data/motionData.csv', 'w')

for ii in range(1, len(dataPoints['xPos'])):
    if dataPoints['index'][ii-1] +1 != dataPoints['index'][ii]:
        print(f"\nData for motion {dataPoints['index'][ii-1] +1} not found")
        continue

    prevMotion = [ dataPoints['xPos'][ii-1], dataPoints['yPos'][ii-1], dataPoints['zPos'][ii-1], dataPoints['xRot'][ii-1], dataPoints['yRot'][ii-1], dataPoints['zRot'][ii-1] ]
    currMotion = [ dataPoints['xPos'][ii], dataPoints['yPos'][ii], dataPoints['zPos'][ii], dataPoints['xRot'][ii], dataPoints['yRot'][ii], dataPoints['zRot'][ii] ]


    fooSel = dataPoints['motionSel'][ii]
    print(f"\n\n\nMotion: {fooSel}")
    outMotion = getMotionBetween(prevMotion, currMotion)
    # continue

    if fooSel in 'fbrl':
        motionPts[fooSel]['xPos'].append(outMotion[0])
        motionPts[fooSel]['yPos'].append(outMotion[1])
        motionPts[fooSel]['zPos'].append(outMotion[2])
        motionPts[fooSel]['xRot'].append(outMotion[3])
        motionPts[fooSel]['yRot'].append(outMotion[4])
        motionPts[fooSel]['zRot'].append(outMotion[5])
        motionPts[fooSel]['motionDur'].append(dataPoints['motionDur'][ii])

        fooMot = motionPts[fooSel]

        print(f"   motionDur:{fooMot['motionDur'][-1]}")
        print(f"   xPos:{fooMot['xPos'][-1]}")
        print(f"   yPos:{fooMot['yPos'][-1]}")
        print(f"   zPos:{fooMot['zPos'][-1]}")
        print(f"   xRot:{fooMot['xRot'][-1]}")
        print(f"   yRot:{fooMot['yRot'][-1]}")
        print(f"   zRot:{fooMot['zRot'][-1]}")


        # moveCSV.write("motionSel, motionDur, xPos, yPos, zPos, xRot, yRot, zRot\n")
        moveCSV.write(f"{fooSel},{fooMot['motionDur'][-1]},{outMotion[0]},{outMotion[1]},{outMotion[2]},{outMotion[3]},{outMotion[4]},{outMotion[5]},\n")

moveCSV.close()





colSelection = {
    'f': 'green',
    'b': 'blue',
    'l': 'red',
    'r': 'orange',
}




if DO_PLOT_3D:
    for fooSel in 'fbrl':
        plot3D.setColor( colSelection[fooSel] )
        
        for ii in range(len(motionPts[fooSel]['xPos'])):
            plot3D.plotPosTransform(
                (motionPts[fooSel]['xPos'][ii], motionPts[fooSel]['yPos'][ii], motionPts[fooSel]['zPos'][ii], 
                motionPts[fooSel]['xRot'][ii], motionPts[fooSel]['yRot'][ii], motionPts[fooSel]['zRot'][ii],)         
            )
    plot3D.showPlot()


if not DO_PLOT_3D:
    if False:
        for fooSel in 'fbrl':
            fooCol = colSelection[fooSel]
            plt.scatter(motionPts[fooSel]['yPos'], motionPts[fooSel]['zRot'], color=fooCol)
    
        plt.xlabel('yPos')
        plt.ylabel('zRot')
        plt.show()



    if False:
        plt.scatter(motionPts['f']['motionDur'], motionPts['f']['yPos'], color=colSelection['f'])
        # plt.scatter(motionPts['b']['motionDur'], motionPts['b']['yPos'], color=colSelection['b'])
    
        plt.xlabel('Duration (mS)')
        plt.ylabel('Forward Motion (mm)')
        plt.show()

    if True:
        plt.scatter(motionPts['r']['motionDur'], motionPts['r']['zRot'], color=colSelection['r'])
        plt.scatter(motionPts['l']['motionDur'], motionPts['l']['zRot'], color=colSelection['l'])

        xVals = np.array(motionPts['r']['motionDur'])
        A, B = np.polyfit(xVals, motionPts['r']['zRot'], 1)
        plt.plot(xVals, A*xVals + B, label=f"Right Turn Y={A}*X + {B}", color=colSelection['r'])


        print(f"Right A:{A}")
        print(f"Right B:{B}")

        xVals = np.array(motionPts['l']['motionDur'])
        A, B = np.polyfit(xVals, motionPts['l']['zRot'], 1)
        plt.plot(xVals, A*xVals + B, label=f"Left Turn Y={A}*X + {B}", color=colSelection['l'])

        print(f"Left A:{A}")
        print(f"Left B:{B}")

        plt.xlabel('Duration (mS)')
        plt.ylabel('Z Rotation (Rad)')
        plt.legend()
        plt.show()

# print(motionPts)
