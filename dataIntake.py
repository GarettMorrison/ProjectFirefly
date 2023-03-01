import serial
import sys
import os
import time
import struct
import math as m
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import imageio

from py.serialCommunication import listPorts, roverSerial
from py.webcam import webcam, adjacentImages

# Constant values for LEDs
LED_COUNT = 18
LED_X = np.array( [ 112.5833025, 91.92388155, 65, 32.5, 45.96194078, 56.29165125, -56.29165125, -45.96194078, -32.5, -65, -91.92388155, -112.5833025, -56.29165125, -45.96194078, -32.5, 32.5, 45.96194078, 56.29165125, ] )
LED_Y = np.array( [ 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, ] )
LED_Z = np.array( [ 0, 0, 0, -56.29165125, -79.60841664, -97.5, -97.5, -79.60841664, -56.29165125, 0, 0, 0, 97.5, 79.60841664, 56.29165125, 56.29165125, 79.60841664, 97.5, ] )

# LED_X = np.array( [ 225.166605, 183.8477631, 130, 65, 91.92388155, 112.5833025, -112.5833025, -91.92388155, -65, -130, -183.8477631, -225.166605, -112.5833025, -91.92388155, -65, 65, 91.92388155, 112.5833025 ] )
# LED_Y = np.array( [ 130, 183.8477631, 225.166605, 225.166605, 183.8477631, 130, 130, 183.8477631, 225.166605, 225.166605, 183.8477631, 130, 130, 183.8477631, 225.166605, 225.166605, 183.8477631, 130 ] )
# LED_Z = np.array( [ 0, 0, 0, -112.5833025, -159.2168333, -195, -195, -159.2168333, -112.5833025, 0, 0, 0, 195, 159.2168333, 112.5833025, 112.5833025, 159.2168333, 195 ] )


LED_EXCLUSION = [
    [0,1,2], 
    [3,4,5], 
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
]
LED_ARRAY = [LED_X, LED_Y, LED_Z]


listPorts()
roverComms = roverSerial()

webcamComms = webcam(LED_ARRAY, roverComms, LED_EXCLUSION)
webcamComms.updateDisplay()

while True:
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        exit()
    elif k==-1:  # Normally -1 returned,so don't print it
        webcamComms.takePhoto()
        webcamComms.updateJustImg()

    elif k != 32: # If char is not space bar
        print(k)

    else: # Char is space bar, take data pt
        webcamComms.clearData()
        startImg = webcamComms.readImage()
        outDict = webcamComms.readVectors()


        # Find filename to pickle data too
        existingFiles = os.listdir('data/')
        fileInd = 0
        while True:
            fileName = f"run_{fileInd}.pkl"
            if fileName in existingFiles: 
                fileInd += 1
            else:
                break

        # Pickle dictionary
        outFile = open("data/"+fileName, 'wb')
        pkl.dump(outDict, outFile)
        outFile.close()

        # Save image
        cv2.imwrite(f"data/photo_{fileInd}.png", startImg)

        print(f"\nFound {len(outDict['index'])}/{len(LED_X)} positions")
        print(f"Located: {np.sort(outDict['index'])}\n")

        notFound = np.setdiff1d( np.arange(LED_COUNT), outDict['index'] )
        print(f"Not found: {np.sort(notFound)}\n")

        failures = webcamComms.getFails()

        for ii in np.sort(notFound):
            print(f"{ii}:")

            for attempt in failures[ii]:
                print(f"     {attempt[0].ljust(20, ' ')}  {str(attempt[1]).ljust(5, ' ')}  {str(attempt[2]).ljust(5, ' ')}  {str(attempt[3]).ljust(5, ' ')}")
