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

import serial.tools.list_ports

def listPorts():
    ports = serial.tools.list_ports.comports()
    print("Comports Available:")
    for port, desc, hwid in sorted(ports): print("{}: {} [{}]".format(port, desc, hwid))



class roverSerial:
    def __init__(self, _portName):
        self.roverID = ''
        self.portName = _portName

        self.ser = serial.Serial(self.portName, baudrate=9600, timeout=0.5, write_timeout=0.5)
        time.sleep(1.5)

    def isStatusOkay(self):
        if self.ser.isOpen(): return True
        return False

    def setLED(self, pos, val):
        numSet = [1, pos, val, 0]
        self.sendBytes(numSet)

    def doMotion(self, direction, duration):
        if type(direction) == str:
            if direction == 'f' or direction == 'F': direction = 1
            elif direction == 'b' or direction == 'B': direction = 2
            elif direction == 'r' or direction == 'R': direction = 4
            elif direction == 'l' or direction == 'L': direction = 8

        numSet = [2, direction, m.floor(duration/256), m.floor(duration)%256]
            
        # self.sendBytes(numSet)
        
        checkSum_sent = sum(numSet)%256
        byteSend = bytes(numSet + [checkSum_sent])
        self.ser.write(byteSend)
        self.ser.flush()


    def doCelebration(self):
        numSet = [4, 0, 0, 0]
        self.sendBytes(numSet)

    # Read sensor data from rover
    # Data is received as uint32_t list
    def getSensorData(self):
        readBytes = b''

        while len(readBytes) < 6:
            numSet = [8, 0, 0, 0]
            self.sendBytes(numSet)
            readBytes = self.ser.read(6)
        
        readData = np.frombuffer(readBytes, dtype=np.uint16, )

        # readData = np.zeros((3), dtype=np.uint16)


        # for ii in range(3):
        #     readData[ii] = 256*readBytes[ii*2]
        #     readData[ii] += readBytes[ii*2 +1]

        return(readData)


    # Read 8 Character ID from rover
    # Data is received as 8 bytes
    def getRoverID(self):    
        numSet = [16, 0, 0, 0]

        checkSum_sent = sum(numSet)%256
        byteSend = bytes(numSet + [checkSum_sent])

        self.ser.write(byteSend)
        self.ser.flush()

        checkSum = self.ser.read(1)
        print(f"   Checksum:{checkSum} vs sent:{bytes([checkSum_sent])}")

        if len(checkSum) == 0: return('N')
        if checkSum != bytes([checkSum_sent]): return('C')

        readBytes = self.ser.read(6)

        if len(readBytes) < 6: return('L')

        self.RoverID = readBytes.decode()
        
        return(self.RoverID)

        
    def sendBytes(self, numSet):
        checkSum_sent = sum(numSet)%256

        byteSend = bytes(numSet + [checkSum_sent])

        # self.ser.reset_input_buffer()

        setAttempts = 0
        while True:
            setAttempts += 1
            self.ser.write(byteSend)
            self.ser.flush()
            # print(f"\nSending {str(numSet).rjust(15, ' ')} -> {' '.join('{:02x}'.format(x) for x in byteSend)}")
            
            checkSum_read = self.ser.read()
            # print(f"read:{checkSum_read}")
            if(len(checkSum_read) == 0): 
                # print(f"   len(checkSum_read) == 0, checkSum={checkSum_read}")  
                continue

            checkSum_read = int.from_bytes(checkSum_read, 'big')

            if checkSum_read == checkSum_sent:
                break
            # else:
            #     print(f"   {checkSum_read} != {checkSum_sent}")
            # if self.ser.in_waiting > 0: print(f"Dumped {' '.join('{:02x}'.format(x) for x in self.ser.read(self.ser.in_waiting))}")
            if self.ser.in_waiting > 0: self.ser.read(self.ser.in_waiting)








if __name__ == "__main__":
    listPorts()
    exit()



import roverConfig as rc

portConnectedList = {}
portCheckedList = []

connectedRovers = []



def initPortConnection(portName):
    print("\nAttempting to connect to port {}".format(portName))

    try:
        newSerialConn = roverSerial(portName)
        print(f"   Requesting ID")
        newRoverID = newSerialConn.getRoverID()
    except:
        print(f"   Port not set up successfully, bailing")
        portCheckedList.append(portName)
        return

    print(f"   Received ID: {newRoverID}")

    # If received ID has configuration available, save to connectedRovers
    if newRoverID in rc.rover_configSet:
        print(f"   Connected to config set {newRoverID}!")

        connectedRovers.append(newRoverID)
        rc.rover_configSet[newRoverID]['SerialComms'] = newSerialConn

    portConnectedList[portName] = newSerialConn
    portCheckedList.append(portName)



def initPortConnections():

    ports = serial.tools.list_ports.comports()
    for portName, desc, hwid in sorted(ports): 
        if portName not in portConnectedList:
            initPortConnection(portName)