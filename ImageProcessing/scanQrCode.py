import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt 

cap = cv.VideoCapture(0)

def QR_decode(im) : 

    decodedObjects = pyzbar.decode(im)   

    return decodedObjects

def display(im, decodedObjects):

    for decodedObject in decodedObjects: 

        points = decodedObject.polygon

        if len(points) > 4 : 

          hull = cv.convexHull(np.array([point for point in points], dtype=np.float32))
          hull = list(map(tuple, np.squeeze(hull)))

        else : 

          hull = points

        n = len(hull)

        for j in range(0,n):

          cv.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)

    return im


while(True):

    rec, frame = cap.read()

    frame_gr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    decodedObjects = QR_decode(frame_gr)

    for decodedObject in decodedObjects: 

        points = decodedObject.polygon
     
        if len(points) > 4 : 

          hull = cv.convexHull(np.array([point for point in points], dtype=np.float32))
          hull = list(map(tuple, np.squeeze(hull)))

        else : 
          hull = points
         
        n = len(hull)     

        for j in range(0,n):
          cv.line(frame, hull[j], hull[ (j+1) % n], (255,0,0), 3)

        x = decodedObject.rect.left
        y = decodedObject.rect.top

        print('Type : ', decodedObject.type)
        print('Data : ', decodedObject.data,'/n')

        barCode = str(decodedObject.data)
        
        cv.putText(frame, barCode, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv.LINE_AA) 

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('o'):
        break

cv.destroyAllWindows()
cap.release()