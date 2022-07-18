import numpy as np
import cv2 as cv

Main_image = cv.imread("1.jpg", 0)

test = cv.imread("2.jpg", 0)

w, h = test.shape

result = cv.matchTemplate(Main_image, test, cv.TM_CCOEFF_NORMED)

locations = np.where(result >= 0.5)

for pic in zip(*locations[::-1]):
    cv.rectangle(Main_image, pic, (pic[0]+ h, pic[1]+w), (0,0,255), 3)


cv.imshow("pic" , Main_image)

cv.waitKey(0)

cv.destroyAllWindows()