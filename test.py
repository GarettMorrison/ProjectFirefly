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
import sys

import py.positionFuncs as pf 
import py.plot3D as p3d


import numpy as np

import serial.tools.list_ports


# fooPort = "COM3"
port_A = "COM7"
def listPorts():
    ports = serial.tools.list_ports.comports()
    print("Comports Available:")
    for port, desc, hwid in sorted(ports): print("{}: {} [{}]".format(port, desc, hwid))

listPorts()


print(f"Connecting to port_A:{port_A}")
port_A_ser = serial.Serial(port_A, baudrate=9600, timeout=0.5)
time.sleep(2)


exit()



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
    adjustedWaypointSet.append([adjWaypoint[2], adjWaypoint[0], -pf.motionToZAng(adjWaypoint)])
    
waypointSet = np.array(adjustedWaypointSet)
currWaypointInd = 0
currWaypoint = waypointSet[0]

print(f"\nXYR Waypoints:")
for foo in adjustedWaypointSet: print(f"   {np.array(foo)}")





# Calculate rover position correction
def calculateRoverCorrection(mapPosition):
    global startPos, currWaypointInd, currWaypoint
    # Convert rover relative position to map position (3D -> 2D)
    print(f"mapPosition: ({np.array(mapPosition)})")
    print(f"targetWaypoint: ({np.array(currWaypoint)})")
    print(f"Change to target: ({np.array(currWaypoint)[:2] - np.array(mapPosition)[:2]})")



    # Calculate dist to target
    currError = m.sqrt(pow(currWaypoint[1]-mapPosition[1], 2) + pow(currWaypoint[0]-mapPosition[0], 2)) # Get distance to target

    # If dist is less than 50mm, target is hit! 
    if currError < 50:
        print(f"HIT POINT {currWaypoint}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # roverComms.doCelebration() # You did it little dude wooo pop off

        # Update target to next waypoint
        currWaypointInd += 1
        if currWaypointInd >= len(waypointSet): currWaypointInd = 0
        currWaypoint = waypointSet[currWaypointInd]

        # Send updated target positions to plotters
        # sendTargetPositions()

    print(f"error: {currError}")
    

    # Calculate run duration for new data point
    runDuration = round(currError)
    if(runDuration > 800): runDuration = 800
    if(runDuration < 200): runDuration = 200

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
    print(f"Final Angle Correction:{m.degrees(targAng)}")

    # Calculate duration of turn (mS) to hit target angle
    turnDuration = round(targAng/0.0025)

    print(f"turnDuration:{turnDuration}")
    print(f"runDuration:{runDuration}")
    

    # # Send data to plotter over ZMQ. 
    # byteData_double = np.array(mapPosition, np.double) # Double data is XYR position
    # byteData_int = np.array([netPointIndex, 0, roverID, turnDuration, runDuration], np.int32) # Int data is pointIndex, pointType, roverID, turnduration, runDuration
    # roverPos_socket.send(byteData_double.tobytes() + byteData_int.tobytes() )
    # netPointIndex += 1

    return(turnDuration, runDuration)



for ii in range(1, len(waypointSet)):
    print(f"\n\n\n WAYPOINT {ii}")
    currWaypoint = waypointSet[ii-1]
    calculateRoverCorrection(waypointSet[ii])









exit()


LED_X = np.array( [ 0, 0, 0, 28.14582562, 39.80420832, 48.75, 48.75, 39.80420832, 28.14582562, 0, 0, 0, -48.75, -39.80420832, -28.14582562, -28.14582562, -39.80420832, -48.75 ] )
LED_Y = np.array( [ 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5, 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5, 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5 ] )
LED_Z = np.array( [ -56.29165125, -45.96194078, -32.5, -16.25, -22.98097039, -28.14582562, 28.14582562, 22.98097039, 16.25, 32.5, 45.96194078, 56.29165125, 28.14582562, 22.98097039, 16.25, -16.25, -22.98097039, -28.14582562 ] )



print(f"Loading data/waypointRaws.csv")
fileIn = open("data/waypointRaws.csv", "r")
waypointSet = []
for fooLine in fileIn.readlines():
    splt = fooLine.split(',')
    waypointSet.append([])
    for ii in range(6): waypointSet[-1].append(float(splt[ii]))

    # # Normalize waypoint positions
    # waypointSet[-1] = pf.normalizeMotion(waypointSet[-1])

waypointSet = np.array(waypointSet)
startPos = waypointSet[0]


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')


startPos = waypointSet[0]

    
print(f"\nNorm Points:")
for foo in waypointSet: 
    for bar in pf.normalizeMotion(foo):
        print(str(round(bar, 3)).rjust(8, ' '), end=", ")
    print("")


    
print(f"\nRelative Points, no norm:")
for foo in waypointSet: 
    for bar in pf.normalizeMotion(pf.getMotionBetween(foo, startPos)):
        print(str(round(bar, 3)).rjust(8, ' '), end=", ")
    print("")



print(f"\nRelative Points, with norm:")
for foo in waypointSet: 
    for bar in pf.getMotionBetween(foo, startPos):
        print(str(round(bar, 3)).rjust(8, ' '), end=", ")
    print("")


# print(f"\ntype(plane): {type(plane)}")
# print(f"plane: {plane}")

# normal = plane.normal
# yRotAdjust = np.arctan(normal[2] / normal[0])
# xRotAdjust = np.arctan(-normal[1] / normal[0])

# print(f"xRotAdjust:{xRotAdjust}")
# print(f"yRotAdjust:{yRotAdjust}")


print(f"Z Rotation:")
for foo in waypointSet: 
    fooMotion =  pf.getMotionBetween(foo, startPos)

    zAng = pf.motionToZAng(fooMotion)
    # angle = np.arctan2(outPt[2], outPt[0])
    print(f"   {zAng}")

    print("")



for foo in waypointSet: 
    # fooMotion = foo
    # print(f"fooMotion:{fooMotion}")
    # p3d.plotPosTransform(fooMotion, 100)
    # p3d.plotLED_positions(pf.completeMotion(np.array([LED_X, LED_Y, LED_Z]), fooMotion))

    
    fooMotion = pf.getMotionBetween(foo, startPos)
    # print(f"fooMotion:{fooMotion}")
    p3d.plotPosTransform(fooMotion, 100)
    p3d.plotLED_positions(pf.completeMotion(np.array([LED_X, LED_Y, LED_Z]), fooMotion))

# p3d.showPlot()
