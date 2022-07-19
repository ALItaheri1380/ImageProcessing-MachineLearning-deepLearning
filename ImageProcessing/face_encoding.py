import numpy as np
import face_recognition as fr
import cv2 as cv

video_capture = cv.VideoCapture(0)

ali_image = fr.load_image_file("face.jpg")

face_encoding = fr.face_encodings(ali_image)[0]

known_face_encondings = [face_encoding]

known_face_names = ["ali"]

while True: 

    _, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)

    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:

            name = known_face_names[best_match_index]
            
            print("Hey Baby !!")

        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

        cv.rectangle(frame, (left, bottom -35), (right, bottom), (0, 255, 0), cv.FILLED)

        cv.putText(frame, name, (left + 6, bottom - 6), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
       
    cv.imshow('Webcam_facerecognition', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
