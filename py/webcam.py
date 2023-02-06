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
    def __init__(self):
        # setting up cam
        self.cam = cv2.VideoCapture(0)
        result, self.prevImage = self.cam.read()

        # Set up displays to allow rearranging them before recording
        cv2.imshow('Display', adjacentImages([[self.prevImage, self.prevImage]])) # Show original) # Show original
        cv2.waitKey(1)
        plt.draw()
        # plt.pause(10)
        result, self.prevImage = self.cam.read()

        self.xPos_set = []
        self.yPos_set = []
        self.ptCount_set = []

        foo = 0
        photoCount = 0


    def readImage(self):
        result, self.img = self.cam.read()

        img_subtract = cv2.subtract(self.img, self.prevImage)
            
        imgGrey_sub = cv2.cvtColor(img_subtract, cv2.COLOR_BGR2GRAY)
        ret, img_thresh = cv2.threshold(imgGrey_sub, 150, 255, cv2.THRESH_BINARY)

        contours, hierarchy= cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        
        if len(sorted_contours) > 0:        
            for bar in sorted_contours[1:]:
                cv2.drawContours(img_thresh, bar, -1, 50,5)


            mass_y, mass_x = np.where(img_thresh >= 255)
            
            imgPos_x = (np.average(mass_x) - img_thresh.shape[1]/2) / 668
            imgPos_y = (img_thresh.shape[0]/2 - np.average(mass_y)) / 668        
            imgPos_s = len(mass_x)
            
            self.xPos_set.append(imgPos_x)
            self.yPos_set.append(imgPos_y)
            self.ptCount_set.append(imgPos_s)
            

            print(f"img_x:{round(imgPos_x,2)}   img_y:{round(imgPos_y,2)}   img_s:{round(imgPos_s,2)}")
            

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
                [self.img, cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)],
            ])

            # cv2.imshow('Display', img_thresh) # Show original) # Show original
            cv2.imshow('Display', imgSet)
            
            # Save image
            # imgFileName = f"data/img/LEDPOS_{self.photoCount}_{self.foo}.png"
            # cv2.imwrite(imgFileName, imgSet) 

            cv2.waitKey(1)

        else:
            print('No image found')
        

        return([self.img, img_subtract])
    
    def getData(self):
        return([self.xPos_set, self.yPos_set, self.ptCount_set])