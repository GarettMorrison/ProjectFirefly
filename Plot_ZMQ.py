import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2, suppress=True)

import math as m
import zmq



# Setup ZMQ connection
# nameserver for windows, cat /etc/resolv.conf
NameServer = "172.26.240.1"
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(f"tcp://localhost:5556")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

colSet = [
    'orange',
    'cyan',
    'blue',
]



posFileOut = open("data/positionHistory.csv", "+a")

plt.ion()     # turns on interactive mode
plt.draw()    # now this should be non-blocking
plt.pause(0.1)



def drawHex(fooPos, fooCol, radius = 10):
    X = []
    Y = []
    radius = 10

    for ii in range(7):
        X.append(fooPos[0] + radius*m.sin(fooPos[2] + ii*m.pi*2/6))
        Y.append(fooPos[1] + radius*m.cos(fooPos[2] + ii*m.pi*2/6))
    
    plt.plot(X, Y, color=fooCol)


print("Starting Check Loop")
positionHist = []
waypointList = []
currTarget = []

while True:
    # Attempt to load data
    try:
        data = socket.recv(flags=zmq.NOBLOCK)
    except:
        # No data loaded, abandoning
        plt.pause(0.2)
        continue


    # Load data to arrays
    # Double data is XYR position
    # Int data is pointIndex, pointType, roverID, turnduration, runDuration

    doubleByteLen = np.dtype(np.double).itemsize*3 # Number of bytes in 3 doubles
    print(f"")
    print(f"REC len(data): {len(data)}")
    positionData = np.frombuffer(data[:doubleByteLen], dtype=np.double)
    metaData = np.frombuffer(data[doubleByteLen:], dtype=np.int32)

    print(f"REC positionData: {positionData}")
    print(f"REC metaData: {metaData}")

    
    if metaData[1] == 0: positionHist.append(positionData) # Save position to history if point is rover position
    if metaData[1] == 1: currTarget = positionData # Save position as currTarget if point is target position
    if metaData[1] == 2: waypointList.append(positionData) # Save position to history if point is waypoint position

    # If position is rover data, output to csv
    if metaData[1] == 0:
        for foo in positionData: posFileOut.write(f"{foo},")
        for foo in metaData: posFileOut.write(f"{foo},")
        posFileOut.write(f"\n")
    

    # Data is updated, do plot
    plt.cla()
    plt.axis('equal')


    # Draw other waypoints
    for fooPos in waypointList:
        drawHex(fooPos, (0.0, 0.0, 1.0))

    # Reload plot history
    prevPos = []
    posCount = len(positionHist)
    for posInd in range(len(positionHist)):
        fooPos = positionHist[posInd]
    
        # Calc plotColor
        if posCount - posInd < 4:
            plotColor = ( 1.0, (posInd - (posCount-4))/6, 0.0 )
        else:
            plotColor = ( posInd/(posCount-3), 0.0, 0.0 )
        
        posInd += 1

        # Draw hexagon
        drawHex(fooPos, plotColor)

        # If not first point, draw line
        if len(prevPos) > 0:
            plt.plot([prevPos[0], fooPos[0]], [prevPos[1], fooPos[1]], color=plotColor)
        # Save curr value as previous

        prevPos = fooPos 

    # Draw current target hex after
    if len(currTarget) > 0:
        drawHex(currTarget, (0.0, 1.0, 0.0), 5)
        drawHex(currTarget, (0.0, 1.0, 0.0), 10)
        drawHex(currTarget, (0.0, 1.0, 0.0), 15)



