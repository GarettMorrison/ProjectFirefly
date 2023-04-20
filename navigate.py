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

from py.serialCommunication import listPorts, roverSerial
from py.webcam import webcam, adjacentImages
import py.positionFuncs as pf 


# Fix annoying numpy print formatting
np.set_printoptions(precision=3, suppress=True)

# Constant values for LEDs
LED_COUNT = 18
# LED_X = np.array( [ 112.5833025, 91.92388155, 65, 32.5, 45.96194078, 56.29165125, -56.29165125, -45.96194078, -32.5, -65, -91.92388155, -112.5833025, -56.29165125, -45.96194078, -32.5, 32.5, 45.96194078, 56.29165125, ] )
# LED_Y = np.array( [ 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, ] )
# LED_Z = np.array( [ 0, 0, 0, -56.29165125, -79.60841664, -97.5, -97.5, -79.60841664, -56.29165125, 0, 0, 0, 97.5, 79.60841664, 56.29165125, 56.29165125, 79.60841664, 97.5, ] )

LED_X = np.array( [ 0, 0, 0, 28.14582562, 39.80420832, 48.75, 48.75, 39.80420832, 28.14582562, 0, 0, 0, -48.75, -39.80420832, -28.14582562, -28.14582562, -39.80420832, -48.75 ] )
LED_Y = np.array( [ 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5, 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5, 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5 ] )
LED_Z = np.array( [ -56.29165125, -45.96194078, -32.5, -16.25, -22.98097039, -28.14582562, 28.14582562, 22.98097039, 16.25, 32.5, 45.96194078, 56.29165125, 28.14582562, 22.98097039, 16.25, -16.25, -22.98097039, -28.14582562 ] )

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

def getRoverPosition():
    global webcamComms, dataProc_socket
    while(True):
        # Get data from webcam
        print("Checking LEDs")
        webcamComms.clearData()
        startImg = webcamComms.readImage()
        readDict = webcamComms.readVectors()

        # Convert data to numpy arrays
        point_indices = np.array(readDict['index'], dtype = np.uint32)
        point_camXAngle = np.array(readDict['xAng'], dtype = np.double)
        point_camYAngle = np.array(readDict['yAng'], dtype = np.double)
        
        # Send data over socket
        print("Sending data")
        dataProc_socket.send(point_indices.tobytes() + point_camXAngle.tobytes() + point_camYAngle.tobytes())

        # Do data output
        vectSet = np.array([np.ones_like(point_camXAngle), point_camYAngle, point_camXAngle])

        # plt.scatter(np.sin(point_camXAngle), np.sin(point_camYAngle))
        # plt.show()

        # Read response
        recData = dataProc_socket.recv()
        position = np.frombuffer(recData, dtype=np.double)
        
        if len(position) > 3:
            return(position)
        else:
            print("Localization math error, retrying")




# Run calibration if selected to do so
if "cal" in sys.argv or "CAL" in sys.argv:
    print('Calibrating Route!')

    input("Hit enter to calibrate starting position:")

    startPos = getRoverPosition()
    print(f"startPos: ")
    for foo in startPos: print( str(round(foo, 3)).rjust(8, ' '), end = ', ')
    print("")
    
    # Calculate waypoints
    waypointSet = [startPos]
    while True:
        readVal = input("Enter X to save path, otherwise save next point:")

        # Break if X was entered
        if readVal == "X" or readVal == 'x': break
        
        readPos = getRoverPosition()
        waypointSet.append(readPos)
        print(f"Read: ")
        for foo in readPos: print( str(round(foo, 3)).rjust(8, ' '), end = ', ')
        print("")



    # # If more than three waypoints, calculate floor plane from position not angle of starting position
    # if(len(waypointSet) > 2):
    #     from skspatial.objects import Plane, Points

    #     # Get array of just XYZ positions 
    #     pointSet = np.array([pf.getMotionBetween(startPos, foo)[:3] for foo in waypointSet])
    #     plane = Plane.best_fit(pointSet)

    #     print(f"pointSet:")
    #     for foo in waypointSet: print(f"   {pf.getMotionBetween(startPos, foo)}")
    #     print(f"type(plane): {type(plane)}")
    #     print(f"plane: {plane}")

    #     normal = plane.normal
    #     xRotAdjust = np.arctan(normal[2] / normal[0])
    #     yRotAdjust = np.arctan(-normal[1] / normal[0])

    print(f"Saving to data/waypointRaws.csv")
    fileOut = open("data/waypointRaws.csv", "w")
    for foo in waypointSet: 
        for bar in foo: fileOut.write(f"{bar},")
        fileOut.write(f"\n")
    fileOut.close()

    print("Waypoint Calibration Complete!")



# Load calibration data and run waypoint mission
print(f"Loading data/waypointRaws.csv")
fileIn = open("data/waypointRaws.csv", "r")
waypointSet = []
for fooLine in fileIn.readlines():
    splt = fooLine.split(',')
    waypointSet.append([])
    for ii in range(6): waypointSet[-1].append(float(splt[ii]))
    waypointSet[-1] = pf.normalizeMotion(waypointSet[-1])

    print(f"   {waypointSet[-1]}")

startPos = pf.normalizeMotion(waypointSet[0])
adjustedWaypointSet = []
for fooWaypoint in waypointSet:
    adjWaypoint = pf.getMotionBetween(fooWaypoint, startPos)
    adjustedWaypointSet.append([adjWaypoint[2], adjWaypoint[0], pf.motionToZAng(adjWaypoint)])
    
waypointSet = np.array(adjustedWaypointSet)
currWaypointInd = 0
currWaypoint = waypointSet[0]

print(f"\nXYR Waypoints:")
for foo in adjustedWaypointSet: print(f"   {np.array(foo)}")


# ZMQ Communication to broadcast rover position
context = zmq.Context()
roverPos_socket = context.socket(zmq.PUB)
roverPos_socket.bind("tcp://*:5556")


# Track total number of points recorded
netPointIndex = 0
roverID = 0 # Dummy value as only one rover is allowed at a time rn

# Send target plots to Plot_ZMQ
print("Sending ZMQ waypointSet")
def sendTargetPositions():
    global netPointIndex, roverID, waypointSet
    for ii in range(len(waypointSet)):
        if ii == currWaypointInd: waypointType = 1
        else: waypointType = 2

        # Send data to plotter over ZMQ. 
        byteData_double = np.array(waypointSet[ii], np.double) # Double data is XYR position
        byteData_int = np.array([netPointIndex, waypointType, roverID, 0, 0], np.int32) # Int data is pointIndex, pointType, roverID, turnDuration, runDuration
        roverPos_socket.send(byteData_double.tobytes() + byteData_int.tobytes() )
        netPointIndex += 1
        
        time.sleep(0.5)


sendTargetPositions()
sendTargetPositions()


# Calculate rover position correction
def calculateRoverCorrection(relativePos):
    global roverPos_socket, startPos, currWaypointInd, currWaypoint, netPointIndex


    # Convert rover relative position to map position (3D -> 2D)
    print(f"relativePos: ({np.array(relativePos)})")
    mapPosition = [relativePos[2], relativePos[0], pf.motionToZAng(relativePos)]
    print(f"mapPosition: ({np.array(mapPosition)})")
    print(f"targetWaypoint: ({np.array(currWaypoint)})")
    print(f"Change to target: ({np.array(currWaypoint)[:2] - np.array(mapPosition)[:2]})")



    # Calculate dist to target
    currError = m.sqrt(pow(currWaypoint[1]-mapPosition[1], 2) + pow(currWaypoint[0]-mapPosition[0], 2)) # Get distance to target

    # If dist is less than 50mm, target is hit! 
    if currError < 20:
        print(f"HIT POINT {currWaypoint}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        roverComms.doCelebration() # You did it little dude wooo pop off

        # Update target to next waypoint
        currWaypointInd += 1
        if currWaypointInd >= len(waypointSet): currWaypointInd = 0
        currWaypoint = waypointSet[currWaypointInd]

        # Send updated target positions to plotters
        sendTargetPositions()

    print(f"error: {currError}")
    

    # Calculate run duration for new data point
    runDuration = round(currError*5)
    if(runDuration > 600): runDuration = 600
    if(runDuration < 150): runDuration = 150

    # Calculate correction
    # Use atan to get angle in radians 
    targAng = np.arctan2((currWaypoint[0]-mapPosition[0]), (currWaypoint[1]-mapPosition[1]))
    print(f"Raw Target Angle Calc:{m.degrees(targAng)}, Current Angle:{m.degrees(mapPosition[2])}")

    # Subtract current angle from target to convert to error
    targAng -= mapPosition[2]
    
    # targAng += runDuration/2000 # Adjust for motor turning error

    # Adjust to be in range +/- pi
    if targAng > m.pi: targAng -= m.pi*2
    if targAng < -m.pi: targAng += m.pi*2

    if targAng < 0: targAng *= 1.5
    
    print(f"Final Angle Correction:{m.degrees(targAng)}")

    # Calculate duration of turn (mS) to hit target angle
    turnDuration = round(targAng/0.003)

    if abs(targAng) > 0.5:
        runDuration = 0
    

    print(f"turnDuration:{turnDuration}")
    print(f"runDuration:{runDuration}")
    

    # Send data to plotter over ZMQ. 
    byteData_double = np.array(mapPosition, np.double) # Double data is XYR position
    byteData_int = np.array([netPointIndex, 0, roverID, turnDuration, runDuration], np.int32) # Int data is pointIndex, pointType, roverID, turnduration, runDuration
    roverPos_socket.send(byteData_double.tobytes() + byteData_int.tobytes() )
    netPointIndex += 1

    # return(0, 0)
    return(turnDuration, runDuration)



while True:
    print('\n\n')
    # Get current rover position and convert to startPos frame
    currPos = getRoverPosition()
    relativePos = pf.getMotionBetween(currPos, startPos)

    turnDuration, runDuration = calculateRoverCorrection(relativePos)
    
    # Execute turn
    if turnDuration < 0: roverComms.doMotion('r', -turnDuration)
    else: roverComms.doMotion('l', turnDuration)

    # Execute forward drive
    roverComms.doMotion('f', runDuration)

    # Wait for rover to finish motion
    time.sleep(runDuration/1000)    