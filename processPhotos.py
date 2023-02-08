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


LED_BRIGHTNESS = 60


LED_X = np.array( [ 112.5833025, 91.92388155, 65, 32.5, 45.96194078, 56.29165125, -56.29165125, -45.96194078, -32.5, -65, -91.92388155, -112.5833025, -56.29165125, -45.96194078, -32.5, 32.5, 45.96194078, 56.29165125, ] )
LED_Y = np.array( [ 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, ] )
LED_Z = np.array( [ 0, 0, 0, -56.29165125, -79.60841664, -97.5, -97.5, -79.60841664, -56.29165125, 0, 0, 0, 97.5, 79.60841664, 56.29165125, 56.29165125, 79.60841664, 97.5, ] )


listPorts()
roverComms = roverSerial()

webcamComms = webcam()
img = webcamComms.readImage()
cv2.imshow('Display', adjacentImages([img, np.zeros_like(img), np.zeros_like(img)]))

while True:
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        exit()
    elif k==-1:  # normally -1 returned,so don't print it
        img = webcamComms.readImage()
        cv2.imshow('Display', adjacentImages([[img, np.zeros_like(img),], [np.zeros_like(img), np.zeros_like(img)]]))

    elif k != 32:
        print(k) # else print its value

    else:
        for ii in range(18):
            roverComms.setLED(ii, [LED_BRIGHTNESS, LED_BRIGHTNESS, LED_BRIGHTNESS] )
            time.sleep(0.2)
            img, subtract = webcamComms.readAndProcImage(ii)
            roverComms.setLED(ii, [0, 0, 0] )
            
        # for foo in webcamComms.getData():
        #     print(foo)


        camData = webcamComms.getData()

        led_indices = np.array(camData[3])

        outDict = {
            'xPix': camData[0],
            'yPix': camData[1],
            'size': camData[2],
            'index': camData[3],
            'LED_X': LED_X[led_indices],
            'LED_Y': LED_Y[led_indices],
            'LED_Z': LED_Z[led_indices],
        }


        existingFiles = os.listdir('data/')

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


        # webcamComms.saveGif("data/"+fileName.split('.',)[0]+".gif")


        ptCount = len(camData[0])
        plotColor = []
        for ii in range(ptCount):
            colVal = 1.0*ii/ptCount
            plotColor.append([1.0-colVal, 0.0, 0.0])


        plt.scatter(camData[0], camData[1], color=plotColor)
        plt.ion()
        plt.pause(1)

        # webcamComms.displayThresh()

        webcamComms.clearData()