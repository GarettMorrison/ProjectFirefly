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
from copy import deepcopy


TRESH_MIN = 120


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







class webcam:
    def clearData(self):
        self.xPos_set = []
        self.yPos_set = []
        self.ptCount_set = []
        self.ledIndices = []
        
        self.imagelist = []
        self.threshlist = []


    def __init__(self):
        # setting up cam
        self.cam = cv2.VideoCapture(0)
        result, self.prevImage = self.cam.read()

        # Set up displays to allow rearranging them before recording
        # cv2.imshow('Display', adjacentImages([[self.prevImage, self.prevImage]])) # Show original) # Show original
        # cv2.waitKey(1)
        # plt.draw()
        self.clearData()

        foo = 0
        photoCount = 0



    def readImage(self):
        result, self.img = self.cam.read()
        return(self.img)
    
    def readAndProcImage(self, ledIndex):
        self.readImage()

        img_subtract = cv2.subtract(cv2.GaussianBlur(self.img, (5,5), 0), cv2.GaussianBlur(self.prevImage, (5,5), 0))
            
        imgGrey_sub = cv2.cvtColor(img_subtract, cv2.COLOR_BGR2GRAY)
        ret, img_thresh = cv2.threshold(imgGrey_sub, TRESH_MIN, 255, cv2.THRESH_BINARY)

        contours, hierarchy= cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        
        if len(sorted_contours) > 0:        
            for bar in sorted_contours[1:]:
                cv2.drawContours(img_thresh, bar, -1, 50, 5)

            mass_y, mass_x = np.where(img_thresh >= 255)
            
            imgPos_x = (np.average(mass_x) - img_thresh.shape[1]/2) / 668
            imgPos_y = (img_thresh.shape[0]/2 - np.average(mass_y)) / 668        
            imgPos_s = len(mass_x)
            
            self.xPos_set.append(imgPos_x)
            self.yPos_set.append(imgPos_y)
            self.ptCount_set.append(imgPos_s)
            self.ledIndices.append(ledIndex)
            

            print(f"img_x:{round(imgPos_x,2)}   img_y:{round(imgPos_y,2)}   img_s:{round(imgPos_s,2)}")
            

            # print(f"img_thresh: {type(img_thresh)}     {img_thresh.shape}    {np.array(np.where(img_thresh >= 255)).shape}")
            # self.threshlist.append(img_thresh)

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
                [self.img, np.zeros_like(self.img),],
                [img_subtract, cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)],
            ])

            # cv2.imshow('Display', img_thresh) # Show original) # Show original
            cv2.imshow('Display', imgSet)
            self.imagelist.append(cv2.cvtColor(imgSet, cv2.COLOR_BGR2RGB))
            
            # Save image
            # imgFileName = f"data/img/LEDPOS_{self.photoCount}_{self.foo}.png"
            # cv2.imwrite(imgFileName, imgSet) 

            cv2.waitKey(1)

        else:
            print('No image found')
        

        return([self.img, img_subtract])
    
    def getData(self):
        return([self.xPos_set, self.yPos_set, self.ptCount_set, self.ledIndices])


    def saveGif(self, fileName):
        imageio.mimsave(fileName, self.imagelist, fps=0.5)


    def displayThresh(self):
        threshCompound = np.zeros_like(cv2.cvtColor(self.threshlist[0], cv2.COLOR_GRAY2BGR))
        

        for ii in range(len(self.threshlist)):
            fooColFactor = round(255*ii/len(self.threshlist))
            fooColor = np.array([0, fooColFactor, 255-fooColFactor])
            
            import random
            fooColor = np.array([round(random.random()*200+55), round(random.random()*200+55), round(random.random()*200+55)])

            # np.place(threshCompound, self.threshlist[ii]>=255, fooColor)
            threshCompound[np.where((self.threshlist[ii] >= 255))] = fooColor
            
            # print(f"img_thresh: {np.array(np.where(self.threshlist[ii] >= 255)).shape}      fooColor:{fooColor}")

        
        cv2.imshow('Display', threshCompound)
        cv2.waitKey(1000)
        time.sleep(1000)



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        
        frame = cv2.circle(frame, (round(frame.shape[1]/2), round(frame.shape[0]/2)), round(frame.shape[0]/4), (0, 0, 255), 1)
        frame = cv2.line(frame, (0, round(frame.shape[0]/2)), (frame.shape[1], round(frame.shape[0]/2)), (0, 0, 255), 1)
        frame = cv2.line(frame, (round(frame.shape[1]/2), 0), (round(frame.shape[1]/2), frame.shape[0]), (0, 0, 255), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()