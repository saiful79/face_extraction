import cv2
import os
from PIL import Image
import numpy as np

FaceClassifier =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img="/home/saiful/Desktop/data_set/image/1.JPG"
cap = cv2.VideoCapture(img)
ret,frame = cap.read()
# print(ret,frame)
print(frame.shape)
minisize = (frame.shape[1],frame.shape[0])

miniframe = cv2.resize(frame, minisize)
faces =  FaceClassifier.detectMultiScale(miniframe)
for f in faces:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255))
    #Save just the rectangle faces in SubRecFaces
    sub_face = frame[y:y+h, x:x+w]
    roi_color=frame[y:y+h, x:x+w]
    eyes=eye_cascade.detectMultiScale(sub_face)
    for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),2)
        FaceFileName = "face_" + str(y) + ".jpg"
        cv2.imwrite(FaceFileName, sub_face)
print("done")