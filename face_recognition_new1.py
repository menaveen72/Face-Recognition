# -*- coding: utf-8 -*-
"""
@author: Naveen
"""

import cv2
import numpy as np
import face_recognition as fr


imgTrain = fr.load_image_file("./known_pic/Elon1.jpg")
imgTrain = cv2.cvtColor(imgTrain,cv2.COLOR_BGR2RGB)
imgTest = fr.load_image_file("./unknown_pic/multifaces.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

face_landmarks_list = fr.face_landmarks(imgTest)
print(face_landmarks_list)
FaceLoc = fr.face_locations(imgTrain)[0]
EncodeTrain = fr.face_encodings(imgTrain)[0]
cv2.rectangle(imgTrain, (FaceLoc[3],FaceLoc[0]),(FaceLoc[1],FaceLoc[2]), (255,0,0),2 )
 
FaceLocTest = fr.face_locations(imgTest)[0]
EncodeTest = fr.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (FaceLocTest[3],FaceLocTest[0]),(FaceLocTest[1],FaceLocTest[2]), (255,0,0),2 )



results = fr.compare_faces([EncodeTrain], EncodeTest)
faceDis = fr.face_distance([EncodeTrain], EncodeTest)

cv2.putText(imgTest, f"{results} {round(faceDis[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)

print(results,faceDis)







cv2.imshow('Train', imgTrain)
cv2.imshow('Test', imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()


