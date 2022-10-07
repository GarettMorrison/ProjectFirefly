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
ports = serial.tools.list_ports.comports()
print("Comports Available:")
for port, desc, hwid in sorted(ports): print("{}: {} [{}]".format(port, desc, hwid))



def adjacentImages(imgArr): # Place images into grid for easy comparison
    blankImg = np.zeros_like(imgArr)
    # blankImg = np.zeros((imgArr[0][0].shape[0], imgArr[0][0].shape[1], 3))

    rowMax = max([len(ii) for ii in imgArr])
    # rowMax = max([len(ii) for ii in imgArr])
    
    for fooRow in imgArr:
        while (len(fooRow) < rowMax):
            fooRow.append(blankImg)

    imgRows = []
    for fooRow in imgArr:
        concatRow = np.concatenate(fooRow, axis=1)
        # if(len(fooRow) < rowMax):
        #     concatRow = np.concatenate([concatRow] + [blankImg]*(rowMax - len(fooRow)), axis=1)
        
        imgRows.append(concatRow)
    

    return(np.concatenate(imgRows, axis=0))



def edge_filter(img):
    # Filter out noise, get solid particle outline
    kernelDim = 4
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelDim, kernelDim))
    # kernel = np.ones((kernelDim, kernelDim),np.uint8)
    imgOut = cv2.morphologyEx(img, cv2.MORPH_ELLIPSE, kernel)
    return(imgOut)


print('Connecting to ports')
# ser = serial.Serial('COM6', baudrate=9600, timeout=0, parity=serial.PARITY_EVEN, stopbits=1)


ser = serial.Serial('COM3', baudrate=4800, timeout=2)
time.sleep(2)


# setting up cam
cam = cv2.VideoCapture(1)
result, prevImage = cam.read()

# Set up displays to allow rearranging them before recording
cv2.imshow('Display', adjacentImages([[prevImage, prevImage]])) # Show original) # Show original
cv2.waitKey(1)
plt.draw()
# plt.pause(10)

print('Starting loop')


ledSetAttempts = []
def setLED(pos, vals):
    numSet = [pos] + vals
    checkSum_sent = sum(numSet)%256
    byteSend = bytes(numSet + [checkSum_sent])
    
    # print(f"setting {pos} to {vals}, sending {byteSend}")

    setAttempts = 0
    while True:
        setAttempts += 1
        ser.write(byteSend)

        checkSum_read = ser.read()
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
            ser.reset_input_buffer()



rowPos_set = []
heightPos_set = []
xPos_set = []
yPos_set = []
ptCount_set = []

foo = 0
photoCount = 0
while True:
    time.sleep(0.05)
    # Take picture
    result, img = cam.read()


    # Set LED Values
    highVal = 10

    rowPos = foo%3
    heightPos = int(foo/3)%38
    setLED(heightPos + rowPos*38, [0,0,0])

    foo += 23
    
    rowPos = foo%3
    heightPos = int(foo/3)%38
    # print(f"rowPos:{rowPos}, heightPos:{heightPos}")
    setLED(heightPos + rowPos*38, [highVal,highVal,highVal])
    
    

    img_subtract = cv2.subtract(img, prevImage)
    	
    imgGrey_sub = cv2.cvtColor(img_subtract, cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(imgGrey_sub, 150, 255, cv2.THRESH_BINARY)

    contours, hierarchy= cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    
    if len(sorted_contours) > 0:        
        for bar in sorted_contours[1:]:
            cv2.drawContours(img_thresh, bar, -1, 50,5)


        mass_y, mass_x = np.where(img_thresh >= 255)
        
        rowPos_set.append(rowPos)
        heightPos_set.append(heightPos)
        imgPos_x = (np.average(mass_x) - img_thresh.shape[1]/2) / 668
        imgPos_y = (img_thresh.shape[0]/2 - np.average(mass_y)) / 668        
        imgPos_s = len(mass_x)
        
        xPos_set.append(imgPos_x)
        yPos_set.append(imgPos_y)
        ptCount_set.append(imgPos_s)
        

        print(f"foo:{foo}   photoCount:{photoCount}   row:{round(rowPos,2)}   col:{round(heightPos,2)}   img_x:{round(imgPos_x,2)}   img_y:{round(imgPos_y,2)}   img_s:{round(imgPos_s,2)}")
        

        # Plot

        # xPos_set.append(imgPos_x)
        # yPos_set.append(imgPos_y)
        # ptCount_set.append(imgPos_s)
        
        # alphaSet = [foo / max(ptCount_set) for foo in ptCount_set]
        # plt.cla()
        # plt.scatter(xPos_set, yPos_set, alpha=alphaSet, color='orange')
        # plt.draw()
        # plt.pause(0.05)


        
        imgSet = adjacentImages([
            [img, cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)],
        ])

        # cv2.imshow('Display', img_thresh) # Show original) # Show original
        cv2.imshow('Display', imgSet)
        
        # Save image
        imgFileName = f"data/img/LEDPOS_{photoCount}_{foo}.png"
        # cv2.imwrite(imgFileName, imgSet) 

        cv2.waitKey(1)

        
        prevImage = img
        photoCount += 1
        # time.sleep(0.5)

        if foo >= 114: 
            foo -= 114


            saveDict = {
                "row" : rowPos_set,
                "col" : heightPos_set,
                "xAngle" : xPos_set,
                "yAngle" : yPos_set,
                "ptSize" : ptCount_set,
            }

            outFile = open('data/outData.pkl', 'wb')
            pkl.dump(saveDict, outFile)
            outFile.close()

            # plt.show()