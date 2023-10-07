import cv2 as cv
import math as m
import mediapipe as mp
import win32api


def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

good_frames = 0
bad_frames = 0


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


cap = cv.VideoCapture(0)

while True:

        success, image = cap.read()

        fps = cap.get(cv.CAP_PROP_FPS)

        h, w = image.shape[:2]

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        keypoints = pose.process(image)

        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        if offset < 100:
            cv.putText(image, str(int(offset)) + 'in the same direction', (w - 500 , 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (127, 255, 0), 2)

        else:
            cv.putText(image, str(int(offset)) + 'It is not in the same direction', (w - 550 , 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)

        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        cv.circle(image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
        cv.circle(image, (l_ear_x, l_ear_y), 7, (0, 255, 255), -1)

        cv.circle(image, (l_shldr_x, l_shldr_y - 100), 7, (0, 255, 255), -1)
        cv.circle(image, (r_shldr_x, r_shldr_y), 7, (50 , 50 , 255), -1)
        cv.circle(image, (l_hip_x, l_hip_y), 7, (0, 255, 255), -1)
        cv.circle(image, (l_hip_x, l_hip_y - 100), 7, (0, 255, 255), -1)

        if neck_inclination < 40 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1
            
            cv.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (127, 233, 100), 2)
            cv.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (127, 233, 100), 2)

            cv.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (127, 255, 0), 4)
            cv.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (127, 255, 0), 4)
            cv.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (127, 255, 0), 4)
            cv.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), (127, 255, 0), 4)

        else:
            good_frames = 0
            bad_frames += 1

            cv.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)
            cv.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)

            cv.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (50, 50, 255), 4)
            cv.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (50, 50, 255), 4)
            cv.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (50, 50, 255), 4)
            cv.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), (50, 50, 255), 4)

        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames

        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv.putText(image, time_string_good, (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (127, 255, 0), 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv.putText(image, time_string_bad, (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)

        if bad_time > 5:
            win32api.Beep(2000 , 1000)

        cv.imshow('body position', image)
        if cv.waitKey(1) & 0xFF == ord('o'):
            break

cap.release()
cv.destroyAllWindows()