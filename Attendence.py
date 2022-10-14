# -*- coding: utf-8 -*-
"""
@author: harsh
"""


import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

path = "C:\\Users\\harsh\\Desktop\\face_recogn\\face_recognition\\known_pic"
Images = []
classNames = []
my_list = os.listdir(path)
print(my_list)
for cl in my_list:
    curImg = cv2.imread(f'{path}/{cl}')
    Images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findencodings(Images):
    encodelist = []
    for img in Images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        Encode = fr.face_encodings(img)[0]
        encodelist.append(Encode)
    return encodelist

def markAttendence(name):
    with open("C:\\Users\\harsh\\Desktop\\face_recogn\\face_recognition\\attendence.csv",'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        #print(mydatalist)
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

markAttendence("Elon1")       

    


encodelistknown = findencodings(Images)
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0) ,None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurFrames = fr.face_locations(imgS)
    EncodeCurFrame = fr.face_encodings(imgS, faceCurFrames)
    
    for encodeFace , faceloc in zip(EncodeCurFrame, faceCurFrames):
        matches = fr.compare_faces(encodelistknown, encodeFace)
        faceDis = fr.face_distance(encodelistknown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
                   name = classNames[matchIndex]
                   #print(name)
                   y1,x2,y2,x1 = faceloc
                   y1,x2,y2,x1 = y1*4,x2*4 ,y2 *4,x1*4
                   cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                   cv2.rectangle(img,(x1,y1-40),(x2,y2),(0,255,0),cv2.FILLED)
                   cv2.putText(img, name, (x1+3,y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                   markAttendence("Elon1")
                   
    cv2.imshow('webcam',img)
    cv2.waitKey(1)
    
print('Encoding Fnished')
