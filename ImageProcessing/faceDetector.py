import cv2 as cv
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

camera = cv.VideoCapture(0)

detector = FaceDetector()

meshDetector = FaceMeshDetector(10)

while(True):
    tmp , frame = camera.read()
    frame , information = detector.findFaces(frame)
    frame , all_face = meshDetector.findFaceMesh(frame)

    if(information):
        center  = information [0] ["center"]
        cv.circle(frame , center , 10 , (0 , 0 , 0) , cv.FILLED) 
        print(information)

    cv.imshow("Face" , frame)

    if cv.waitKey(1) & 0xFF == ord('o'):
        break

cv.destroyAllWindows()

camera.release()
