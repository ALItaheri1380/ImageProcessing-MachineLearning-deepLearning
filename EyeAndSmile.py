import cv2 as cv

face_case = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_case = cv.CascadeClassifier('haarcascade_eye.xml')
smil_case = cv.CascadeClassifier('haarcascade_smile.xml')

camera = cv.VideoCapture(0)

while(True):

    vain,frame = camera.read()

    frame_gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)

    all_face = face_case.detectMultiScale(frame_gray , 1.3 , 5)

    for (x , y , w , h) in all_face:

        cv.rectangle(frame , (x , y) , (x+w , y+h) , (0 , 0 , 0) , 5)

        frame_gry_roi = frame_gray[y : y + h , x : x + w]

        frame_roi = frame[y : y + h , x : x + w]

        eyes = eyes_case.detectMultiScale(frame_gry_roi)

        for (ex , ey , ew , eh) in eyes:

            cv.rectangle(frame_roi , (ex , ey) , (ex + ew , ey + eh) , (250,0,0) , 5)

        smils = smil_case.detectMultiScale(frame_gry_roi , 1.8 , 20)

        for (sx , sy , sw , sh) in smils:
            cv.rectangle(frame_roi , (sx , sy) , (sx + sw , sy + sh), (0,0,255) , 5) 

    cv.imshow('faces' , frame)    

    if cv.waitKey(1) & 0xFF == ord('o'):
        break

cv.destroyAllWindows()

camera.release()    



