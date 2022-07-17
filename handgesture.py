from time import sleep
from cvzone.HandTrackingModule import HandDetector
import cv2 as cv

camera = cv.VideoCapture(0)

detector = HandDetector(0.5 , 2)

while(True):

    tmp , frame = camera.read()

    hands , frame = detector.findHands(frame)

    if (hands):

        first_hand = hands[0]

        lmlist1 = first_hand["lmList"]

        Len , information , frame = detector.findDistance(lmlist1[9][0:-1] , lmlist1[12][0:-1] , frame) 
        
        print("len is = ", Len)

    cv.imshow('frame' , frame)
    
    if cv.waitKey(1) & 0xFF == ord('o'):

        break

cv.destroyAllWindows()

camera.release()