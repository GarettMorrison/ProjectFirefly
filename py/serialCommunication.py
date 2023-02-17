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
    def __init__(self):
        print('Connecting to ports')
        # ser = serial.Serial('COM6', baudrate=9600, timeout=0, parity=serial.PARITY_EVEN, stopbits=1)

        self.ser = serial.Serial('COM3', baudrate=4800, timeout=2)
        time.sleep(1)

    ledSetAttempts = []
    def setLED(self, pos, vals):
        numSet = [pos] + vals
        checkSum_sent = sum(numSet)%256
        byteSend = bytes(numSet + [checkSum_sent])
        
        # print(f"setting {pos} to {vals}, sending {byteSend}")

        setAttempts = 0
        while True:
            setAttempts += 1
            self.ser.write(byteSend)

            checkSum_read = self.ser.read()
            # print(f"read:{checkSum_read}")
            if(len(checkSum_read) == 0): 
                # print(f"   len(checkSum_read) == 0, checkSum={checkSum_read}")  
                continue

            checkSum_read = int.from_bytes(checkSum_read, 'big')

            if checkSum_read == checkSum_sent:
                # print(f"   {checkSum_read} == {checkSum_sent}")

                # ledSetAttempts.append(setAttempts)
                # if len(ledSetAttempts) > 100: del ledSetAttempts[0]
                # print(f"ledSetAttempts average:{ sum(ledSetAttempts) / len(ledSetAttempts) }")
                break
            else:
                # print(f"   {checkSum_read} != {checkSum_sent}")
                self.ser.reset_input_buffer()

