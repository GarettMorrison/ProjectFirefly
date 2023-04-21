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
from copy import deepcopy

import py.serialCommunication as comms
from py.webcam import webcam, adjacentImages
import py.positionFuncs as pf 

import roverConfig as rc
import SteeringML_ZMQ as SML

# Fix annoying numpy print formatting
np.set_printoptions(precision=3, suppress=True)

ACCEPTABLE_WAYPOINT_DIST = 50 # Allowable error for waypoint to be considered found


# Rover Communication
comms.initPortConnections()

# comms.initPortConnection('COM7') 
# # comms.initPortConnection(' COM8')
# 
# comms.initPortConnection(' COM9')
# comms.initPortConnection(' COM10')



if len(comms.connectedRovers) < 1: 
    print(f"No rovers detected, bailing")
    exit()



while False:
    for roverName in comms.connectedRovers: 
        fooSerialComms = rc.rover_configSet[roverName]['SerialComms']    
        fooSerialComms.doMotion('f', 1500)
        print(f"TESTING Forward {roverName}")
    time.sleep(1.5)

    for roverName in comms.connectedRovers: 
        fooSerialComms = rc.rover_configSet[roverName]['SerialComms']  
        fooSerialComms.doMotion('l', 1500)
        print(f"TESTING Left {roverName}")

    time.sleep(1.5)

        
    for roverName in comms.connectedRovers: 
        fooSerialComms = rc.rover_configSet[roverName]['SerialComms']  
        fooSerialComms.doMotion('r', 1500)
        print(f"TESTING Right {roverName}")

    time.sleep(1.5)





# Webcam communication
webcamComms = webcam()
webcamComms.updateDisplay()

# ZMQ Communication to WSL Data processing script
context = zmq.Context()
dataProc_socket = context.socket(zmq.REQ)
dataProc_socket.bind("tcp://*:5555")

# ZMQ Communication to broadcast rover position
context = zmq.Context()
roverPos_socket = context.socket(zmq.PUB)
roverPos_socket.bind("tcp://*:5556")

# ZMQ Communication to broadcast rover motion
context = zmq.Context()
roverMove_socket = context.socket(zmq.PUB)
roverMove_socket.bind("tcp://*:5557")

previousRoverMapPos = {}
previousRoverMoveDurs = {}

# ZMQ Communication to broadcast absolute position
# context = zmq.Context()
# rawPos_socket = context.socket(zmq.PUB)
# rawPos_socket.bind("tcp://*:5558")


# ZMQ Communication to receive dead reckoning control values from 
# nameserver for windows, cat /etc/resolv.conf
NameServer = rc.LOCALHOST_NAME_SERVER
context = zmq.Context()
zmqML_socket = context.socket(zmq.SUB)
zmqML_socket.connect(f"tcp://{NameServer}:5558")
zmqML_socket.setsockopt_string(zmq.SUBSCRIBE, "")



def getRoverPosition(fooRoverName):
    global webcamComms, dataProc_socket
    while(True):
        # Get data from webcam
        print(f"   Checking LEDs for {fooRoverName}")
        startImg = webcamComms.readImage()
        readDict = webcamComms.readVectors(fooRoverName) # Call readVectors using current rover ID

        # Convert data to numpy arrays

        roverIndex = rc.rover_indexByName[fooRoverName].to_bytes(1, 'big')
        point_indices = np.array(readDict['index'], dtype = np.uint32)
        point_camXAngle = np.array(readDict['xAng'], dtype = np.double)
        point_camYAngle = np.array(readDict['yAng'], dtype = np.double)

        if len(point_indices) < 5:
            print(f"   Not enough points found for {fooRoverName}, continuing")
            return([1])
        
        # Send data over socket
        print(f"   Requesting data processing for rover[{roverIndex}]: {fooRoverName}")
        dataProc_socket.send(roverIndex +  point_indices.tobytes() + point_camXAngle.tobytes() + point_camYAngle.tobytes())

        # Do data output
        # vectSet = np.array([np.ones_like(point_camXAngle), point_camYAngle, point_camXAngle])

        # plt.scatter(np.sin(point_camXAngle), np.sin(point_camYAngle))
        # plt.show()

        # Read response
        recData = dataProc_socket.recv()
        position = np.frombuffer(recData, dtype=np.double)

        # rawPos_socket.send(roverIndex + position.tobytes())
        
        if len(position) > 3:
            return(position)
        else:
            print(f"Localization math error for {fooRoverName}, continuing")
            return([2])


# Call getRoverPosition() until the position converges successfully
def getRoverPosRepeating(fooRoverName):
    fooPos = [1]
    while len(fooPos) < 6:
        fooPos = getRoverPosition(fooRoverName)
    return(fooPos)



# Run calibration if selected to do so
if "cal" in sys.argv or "CAL" in sys.argv:
    calRoverName =  comms.connectedRovers[0]
    print(f"Calibrating Route using {calRoverName}!")
    
    input("Hit enter to calibrate starting position:")

    startPos = getRoverPosRepeating(calRoverName)
    
    print(f"startPos: ")
    for foo in startPos: print( str(round(foo, 3)).rjust(8, ' '), end = ', ')
    print("")
    
    # Calculate waypoints
    waypointSet = [startPos]
    while True:
        readVal = input("Enter X to save path, otherwise save next point:")

        # Break if X was entered
        if readVal == "X" or readVal == 'x': break
        
        readPos = getRoverPosRepeating(calRoverName)
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

# Process waypoints, get starting position
startPos = pf.normalizeMotion(waypointSet[0])
adjustedWaypointSet = []
for fooWaypoint in waypointSet: 
    # Convert 3D camera pos to 2D map position
    adjWaypoint = pf.getMotionBetween(fooWaypoint, startPos)
    adjustedWaypointSet.append([adjWaypoint[2], adjWaypoint[0], pf.motionToZAng(adjWaypoint)])

# Setup position sets for robots
waypointSet = np.array(adjustedWaypointSet) # Convert to np array 
currWaypointIndSet = {}
currWaypointSet = {}
for fooName in rc.rover_configSet:
    currWaypointIndSet[fooName] = 0
    currWaypointSet[fooName] = waypointSet[0]


print(f"\nXYR Waypoints:")
for foo in adjustedWaypointSet: print(f"   {np.array(foo)}")


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
        byteData_int = np.array([netPointIndex, waypointType, roverID, 0, 0, 0, 0, 0], np.int32) # Int data is pointIndex, pointType, roverID, turnDuration, runDuration
        roverPos_socket.send(byteData_double.tobytes() + byteData_int.tobytes() )
        netPointIndex += 1
        
        time.sleep(0.5)


# sendTargetPositions()
# sendTargetPositions()


# Calculate rover position correction
def calculateRoverCorrection(relativePos, fooRoverName):
    global roverPos_socket, startPos, currWaypointIndSet, currWaypointSet, netPointIndex, zmqML_socket

    # # Read updated steering data from SteeringML_ZMQ if available
    # while(True):
        # try:
        #     print(f"   Checking steeringML_socket")
        #     zmqData = steeringML_socket.recv(flags=zmq.NOBLOCK)
        # except:
        #     print(f"   No SteeringML Data")
        #     # No data loaded, skipping
        #     break
        
        # roverID_SML = np.frombuffer(zmqData[:2])[0]
        # dataSelection_SML = np.frombuffer(zmqData[2:4])[0]
        # steeringCalValues = np.array(zmqData[4:], dtype=np.double).tobytes()
        # print(f"   Received SteeringML Data:")
        # print(f"      roverID_SML: {roverID_SML}")
        # print(f"      dataSelection_SML: {dataSelection_SML}")
        # print(f"      steeringCalValues: {steeringCalValues}")
        # roverName_SML = rc.rover_nameByIndex[roverID_SML]
        # rc.rover_configSet[roverName_SML]['SteeringVals'][dataSelection_SML] = steeringCalValues


    dataFileName = "data/roverSteeringCal.pkl"
    try:
        file = open(dataFileName, 'rb')
        rc.rover_steeringVals = pkl.load(file)
        file.close()
    except:
        print(f"Unable to load {dataFileName}")
        

    currWaypointInd = currWaypointIndSet[fooRoverName]
    currWaypoint = currWaypointSet[fooRoverName]
    # Convert rover relative position to map position (3D -> 2D)
    mapPosition = [relativePos[2], relativePos[0], pf.motionToZAng(relativePos)]
    print(f"   mapPosition: ({np.array(mapPosition)})")

    # Calculate dist to target
    currError = m.sqrt(pow(currWaypoint[1]-mapPosition[1], 2) + pow(currWaypoint[0]-mapPosition[0], 2)) # Get distance to target

    # If dist is less than 50mm, target is hit! 
    if currError < ACCEPTABLE_WAYPOINT_DIST:
        print(f"   HIT POINT {currWaypoint}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        rc.rover_configSet[roverName]['SerialComms'].doCelebration() # You did it little dude wooo pop off

        # Update target to next waypoint
        currWaypointInd += 1
        if currWaypointInd >= len(waypointSet): currWaypointInd = 0
        currWaypointIndSet[fooRoverName] = currWaypointInd
        currWaypointSet[fooRoverName] = waypointSet[currWaypointInd]

        print(f"   Now targeting waypoint {currWaypointIndSet[fooRoverName]} : {currWaypointSet[fooRoverName]}")
        # Send updated target positions to plotters
        # sendTargetPositions()

    print(f"   error: {currError}")


    # Calculate motion durations from SML data
    turnDuration, runDuration = SML.calcMotionDurs( mapPosition, currWaypoint, rc.rover_steeringVals[fooRoverName] )

    # # Calculate correction
    # # Use atan to get angle in radians 
    # targAng = np.arctan2((currWaypoint[0]-mapPosition[0]), (currWaypoint[1]-mapPosition[1]))
    # print(f"   Raw Target Angle Calc:{m.degrees(targAng)}, Current Angle:{m.degrees(mapPosition[2])}")

    # # Subtract current angle from target to convert to error
    # targAng -= mapPosition[2]
    
    # # targAng += runDuration/2000 # Adjust for motor turning error

    # # Adjust to be in range +/- pi
    # if targAng > m.pi: targAng -= m.pi*2
    # if targAng < -m.pi: targAng += m.pi*2
    
    # print(f"   Final Angle Correction:{m.degrees(targAng)}")

    # # Calculate duration of turn (mS) to hit target angle
    # # turnDuration = round(targAng * 40)
    # turnDuration = round(targAng * TURN_DUR_FACTOR)

    
    # print(f"   turnDuration:{turnDuration}")
    # print(f"   runDuration:{runDuration}")
    
    sensorData = rc.rover_configSet[roverName]['SerialComms'].getSensorData() # Load sensor data to send
    roverID = rc.rover_indexByName[fooRoverName]

    # Rover Map Position over ZMQ. 
    byteData_double = np.array(mapPosition, np.double) # Double data is XYR position
    byteData_int = np.array([netPointIndex, 0, roverID, turnDuration, runDuration, sensorData[0], sensorData[1], sensorData[2]], np.int32) # Int data is pointIndex, pointType, roverID, turnduration, runDuration
    roverPos_socket.send(byteData_double.tobytes() + byteData_int.tobytes() )

    # Send Rover Move Data over ZMQ
    if roverName in previousRoverMapPos:
        mapPosPair = np.array(previousRoverMapPos[roverName] + mapPosition, np.double)
        moveDurs = np.array([netPointIndex, 0, roverID, previousRoverMoveDurs[roverName][0], previousRoverMoveDurs[roverName][1], sensorData[0], sensorData[1], sensorData[2]], np.int32)
        roverMove_socket.send(mapPosPair.tobytes() + moveDurs.tobytes())

    previousRoverMapPos[roverName] = deepcopy(mapPosition)
    previousRoverMoveDurs[roverName] = [turnDuration, runDuration]
    
    netPointIndex += 1

    print(f"   turn&run duration: ({turnDuration}, {runDuration})")

    # return(0, 0)
    return(turnDuration, runDuration)



while True:
    print('\n\n')
    for roverName in rc.rover_configSet:
        if not 'SerialComms' in rc.rover_configSet[roverName]:
            continue
        

        print(f"\nRover[{rc.rover_indexByName[roverName]}]: {roverName}")
        fooSerialComms = rc.rover_configSet[roverName]['SerialComms']
        
        # Get current rover position
        currPos = getRoverPosition(roverName)
        if len(currPos) < 6: continue # If returned array has len < 6 position was not found
        print(f"   {roverName} Absolute Pos: {currPos}")
        

        # convert to startPos frame
        relativePos = pf.getMotionBetween(currPos, startPos)

        turnDuration, runDuration = calculateRoverCorrection(relativePos, roverName)
        


        # Execute turn
        if turnDuration < 0: fooSerialComms.doMotion('r', -turnDuration)
        else: fooSerialComms.doMotion('l', turnDuration)

        time.sleep(abs(turnDuration)/100)    

        # Execute forward drive
        fooSerialComms.doMotion('f', runDuration)

        # Wait for rover to finish motion
        time.sleep(runDuration/500)