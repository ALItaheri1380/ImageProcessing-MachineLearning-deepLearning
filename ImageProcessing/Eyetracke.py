import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

detector = FaceDetector()
Meshdetector = FaceMeshDetector()


camera = cv.VideoCapture(0)
 
while(True):

    _ , frame = camera.read()
    _ , Temp_Fram = camera.read()

    if _ == True:
        image, Box = detector.findFaces(frame)
        image, faces = Meshdetector.findFaceMesh(frame)

        if Box:
            if faces:

                Right_eye_points = np.array([[faces[0][i][0],faces[0][i][1]] for i in [7 , 163 , 144 , 145 , 153 , 154 ,
                155 , 133 , 173 , 157 , 158 , 59 , 160 , 161 , 246]])

                (X,Y,W,H) = cv.boundingRect(Right_eye_points)

                eye_roi = Temp_Fram[Y:Y+H, X:X+W]

                eye_roi_gr = cv.cvtColor(eye_roi, cv.COLOR_BGR2GRAY)

                _, iris = cv.threshold(eye_roi_gr, 40, 255, cv.THRESH_BINARY_INV)

                contours, _ = cv.findContours(iris, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

                if contours:

                    (x,y,w,h) = cv.boundingRect(contours[0])

                    center_x, center_y = x+int(w/2) + X, y+int(h/2)+Y

                    cv.circle(Temp_Fram, (center_x, center_y), 4 , (255,255,0), -1)

                    ix_cntr_e, iy_centr_e = x+int(w/2), y+int(h/2)

                    miss = 8

                    if ix_cntr_e > int(W/2)+miss:
                        print("Right")

                    elif ix_cntr_e < int(W/2)-miss:
                        print("Left")

                    else:
                        print("Center")

        cv.imshow('EYE', Temp_Fram)

        if cv.waitKey(1) & 0xFF == ord('o'):
            break
    else:
        break


cv.destroyAllWindows()
camera.release()