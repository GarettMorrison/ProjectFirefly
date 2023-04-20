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
from random import randint
import zmq

from py.serialCommunication import listPorts, roverSerial
from py.webcam import webcam, adjacentImages
import py.positionFuncs as pf 

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


moveSet = ['l', 'r', 'f', 'b']
moveIndex = 0
motionData = {}

listPorts()
# Rover Communication
roverComms = roverSerial()

# Webcam communication
webcamComms = webcam(LED_ARRAY, roverComms, LED_EXCLUSION)
webcamComms.updateDisplay()

# ZMQ Communication to WSL Data processing script
context = zmq.Context()
dataProc_socket = context.socket(zmq.REQ)
dataProc_socket.bind("tcp://*:5555")

# ZMQ Communication to output rover position
context = zmq.Context()
roverPos_socket = context.socket(zmq.PUB)
roverPos_socket.bind("tcp://*:5556")

def getRoverPosition():
    global webcamComms, dataProc_socket
    # Get data from webcam
    print("Checking LEDs")
    webcamComms.clearData()
    startImg = webcamComms.readImage()
    readDict = webcamComms.readVectors()

    # Convert data to numpy arrays
    point_indices = np.array(readDict['index'], dtype = np.uint32)
    point_zAngle = np.array(readDict['xAng'], dtype = np.double)
    point_yAngle = np.array(readDict['yAng'], dtype = np.double)
    
    # Send data over socket
    print("Sending data")
    dataProc_socket.send(point_indices.tobytes() + point_zAngle.tobytes() + point_yAngle.tobytes())

    # Read response
    recData = dataProc_socket.recv()
    position = np.frombuffer(recData, dtype=np.double)
    return(position)




np.set_printoptions(suppress=True)
startPos = [1310.84671983, 50.7199677, -227.16122983, 0.05799902, -1.01432695, 3.11867319]

while True:
    print('\n\n')
    currPos = getRoverPosition()
    if len(currPos) < 6: 
        print("Localization math error, retrying")
        continue

    relativePos = pf.getMotionBetween(startPos, currPos)

    print(f"currPos: \n   {currPos[:3]} \n   {currPos[3:]}\n")
    print(f"relativePos: \n   {relativePos[:3]} \n   {relativePos[3:]}\n")
    print(f"\n[{relativePos[2]}, {relativePos[1]}, {relativePos[5]}],   #{relativePos}")