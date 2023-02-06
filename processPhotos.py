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

from py.serialCommunication import listPorts, roverSerial
from py.webcam import webcam


LED_X = np.array( [112.5833025, 91.92388155, 65, 0, 0, 0, -112.5833025, -91.92388155, -65, 0, 0, 0] )
LED_Y = np.array( [0, 0, 0, -65, -91.92388155, -112.5833025, 0, 0, 0, 65, 91.92388155, 112.5833025] )
LED_Z = np.array( [68, 94.92388155, 115.5833025, 115.5833025, 94.92388155, 68, 68, 94.92388155, 115.5833025, 115.5833025, 94.92388155, 68] )

roverComms = roverSerial()
webcamComms = webcam()

for ii in range(12):
    roverComms.setLED(ii, [255, 255, 255] )
    time.sleep(0.5)
    img, subtract = webcamComms.readImage()
    roverComms.setLED(ii, [0, 0, 0] )

for foo in webcamComms.getData():
    print(foo)


camData = webcamComms.getData()

outDict = {
    'xPos': camData[0],
    'yPos': camData[1],
    'size': camData[2],
}


existingFiles = os.listdir('data/')
print(existingFiles)

fileInd = 0
while True:
    fileName = f"run_{fileInd}.pkl"
    if fileName in existingFiles: 
        fileInd += 1
    else:
        break


outFile = open("data/"+fileName, 'wb')
pkl.dump(outDict, outFile)
outFile.close()


# input("Finished! Hit any key to exit")