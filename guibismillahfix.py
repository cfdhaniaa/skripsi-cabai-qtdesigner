#Cut Fahrani Dhania | 195150307111016

#Import Library
import cv2
import datetime
import sys
import time
import imutils
import numpy as np
import argparse

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap 
from PyQt5.QtWidgets import QDialog, QApplication, QLabel, QWidget
from PyQt5.uic import loadUi
from collections import deque

#Identify the color values to detect in HSV
buffer = 128
#colorLower = (29, 86, 6)
#colorUpper = (64, 255, 255)
colorLower = np.array([0, 73, 98], np.uint8)
colorUpper = np.array([179, 223, 255], np.uint8)

#Finding red color based on HSV color space
colorLower2 = np.array([0, 153, 128],np.uint8)
colorUpper2 = np.array([255, 255, 255],np.uint8)

#Finding orange color based on HSV color space
#colorLower3 = np.array([8, 128, 179],np.uint8)
#colorUpper3 = np.array([255, 255, 255],np.uint8)

pts = deque(maxlen=buffer)
counter = 0
(dX, dY) = (0, 0)
direction = ""

time.sleep(2.0)

#Create Class
class tehseencode(QDialog):
    def __init__ (self):
        #Create constructor for tehseencode
        super(tehseencode,self).__init__()
        loadUi('untitled3.ui',self)
        self.logic = 0

        #Initialize the buttons and text to run
        self.RUN.clicked.connect(self.RUNClicked)

        self.TEXT.setText('Selamat Datang di Sistem Klasifikasi Kematangan Cabai')
        
        self.STOP.clicked.connect(self.STOPClicked)

        self.QUIT.clicked.connect(self.QUITClicked)
                    
    @pyqtSlot()
#Call RUNClicked Function
    def RUNClicked(self):
        self.logic = 1
        cap = cv2.VideoCapture(1)
        date = datetime.datetime.now()
        out = cv2.VideoWriter('C:/Users/Asus/OneDrive/Documents/SKRIPSI/video/Video_%s%s%sT%s%s%s.mp4' %(date.year,date.month,date.day,date.hour,date.minute,date.second), -1, 20.0, (640,480))
        print('here')
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                self.displayImage(frame, 1)
                cv2.waitKey()
            #Make condition when camera is activate and deactivate
                if (self.logic == 1):
                    out.write(frame)
                    self.TEXT.setText('Mulai melakukan deteksi')

                if (self.logic == 0):
                    self.TEXT.setText('Selesai melakukan deteksi')

                    break

            else:
                print('return not found')

        cap.release()
        cv2.destroyAllWindows()

#Call STOPClicked Function
    def STOPClicked(self):
        self.logic = 0

#Call displayImage Function
    def displayImage(self, frame, window = 1):

            #Convert image to RGB Format
            frame = imutils.resize(frame, width=640, height=480)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, colorLower, colorUpper)
            mask2 = cv2.inRange(hsv, colorLower2, colorUpper2)
            bitwiseOr = cv2.bitwise_or(mask1,mask2)
            #bitwiseOr = cv2.bitwise_or(bitwiseOr, None, iterations=2)
            #bitwiseOr = cv2.bitwise_or(bitwiseOr, None, iterations=2)
        
            contour = cv2.findContours(bitwiseOr.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contour = imutils.grab_contours(contour)
            center = None
            
            #Create a bounding box
            for c in contour:
                rect = cv2.boundingRect(c)
                if rect[2] < 100 or rect[3] < 100: continue
                print(cv2.contourArea(c))
                x,y,w,h = rect
            # Draw the rectangle on the frame
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            # Insert label inside the bounding box
                cv2.putText(frame,'Cabai Matang',(x+w+10,y+h),0,0.3,(0,255,0))
            
                #qformat = QImage.Format_Indexed8

            # if len(frame.shape) == 3:

            #     if (frame.shape[2]) == 4:
            #         qformat = QImage.Format_RGBA888
            # else:
            #         qformat = QImage.Format_RGB888

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width

            #frame = QImage(frame.data, frame.shape[1], frame.shape[0], qformat)
            #frame = frame.rgbSwapped()

            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            #self.imgLabel.setPixmap(QPixmap.fromImage(frame))
            self.imgLabel.setPixmap(QPixmap.fromImage(qImg))

            #Create image for Segmentation
            height_2, width_2= bitwiseOr.shape
            step_2 = 3 * width_2                                                      
            qImg_2 =  QImage(bitwiseOr.data, width_2, height_2, QImage.Format_Grayscale8) 
            self.imgLabel_2.setPixmap(QPixmap.fromImage(qImg_2))

#Call QUITClicked Function
    def QUITClicked(self):
        self.logic = 1
        QUIT.show()

#Count Computation Time
# get the start time
st = time.time()

#find sum to first 1 million numbers
sum_x = 0
for i in range(1000000):
    sum_x += i

# wait for 3 seconds
time.sleep(3)
print('Sum of first 1 million numbers is:', sum_x)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window=tehseencode()
    window.show()
    
    try:
        sys.exit(app.exec_())
    except:
        print('exiting')
