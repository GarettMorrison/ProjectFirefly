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

# fooPort = "COM3"
fooPort = "COM6"
def listPorts():
    ports = serial.tools.list_ports.comports()
    print("Comports Available:")
    for port, desc, hwid in sorted(ports): print("{}: {} [{}]".format(port, desc, hwid))



class roverSerial:
    def __init__(self):
        print('Connecting to ports')
        # ser = serial.Serial('COM6', baudrate=9600, timeout=0, parity=serial.PARITY_EVEN, stopbits=1)

        self.ser = serial.Serial(fooPort, baudrate=9600, timeout=0.5)
        time.sleep(2)

    def setLED(self, pos, val):
        numSet = [1, pos, val, 0]
        self.sendBytes(numSet)

    def doMotion(self, direction, duration):
        if type(direction) == str:
            if direction == 'f' or direction == 'F': direction = 1
            elif direction == 'b' or direction == 'B': direction = 2
            elif direction == 'r' or direction == 'R': direction = 4
            elif direction == 'l' or direction == 'L': direction = 8

        numSet = [2, direction, m.floor(duration/256), duration%256]
        self.sendBytes(numSet)

        
    def sendBytes(self, numSet):
        checkSum_sent = sum(numSet)%256
        byteSend = bytes(numSet + [checkSum_sent])

        # self.ser.reset_input_buffer()

        setAttempts = 0
        while True:
            setAttempts += 1
            self.ser.write(byteSend)
            self.ser.flush()
            print(f"\nSending {str(numSet).rjust(15, ' ')} -> {' '.join('{:02x}'.format(x) for x in byteSend)}")
            
            checkSum_read = self.ser.read()
            # print(f"read:{checkSum_read}")
            if(len(checkSum_read) == 0): 
                print(f"   len(checkSum_read) == 0, checkSum={checkSum_read}")  
                continue

            checkSum_read = int.from_bytes(checkSum_read, 'big')

            if checkSum_read == checkSum_sent:
                break
            else:
                print(f"   {checkSum_read} != {checkSum_sent}")
            if self.ser.in_waiting > 0: print(f"Dumped {' '.join('{:02x}'.format(x) for x in self.ser.read(self.ser.in_waiting))}")

# listPorts()
if __name__ == "__main__":
    listPorts()