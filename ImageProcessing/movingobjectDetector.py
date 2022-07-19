import cv2 as cv

def fit_size(frame , p = 100):

    w = int(frame.shape[1] * p / 100)

    h = int(frame.shape[0] * p / 100)

    d = (w , h)

    return cv.resize(frame , d , cv.INTER_AREA)

camera = cv.VideoCapture('1.mp4')

while(True):
    _ , first_frame = camera.read()

    _ , seconde_frame = camera.read()

    resize_first_frame = fit_size(first_frame)

    resize_seconde_frame = fit_size(seconde_frame)

    different = cv.absdiff(resize_first_frame , resize_seconde_frame)

    convert_gry = cv.cvtColor(different , cv.COLOR_BGR2GRAY)

    blurred_frame = cv.GaussianBlur(convert_gry , (3 , 3) , 1)

    _ , mask = cv.threshold(blurred_frame , 9 , 255 , cv.THRESH_BINARY)

    shapes , _ = cv.findContours(mask , cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)

    for shape in shapes:

        if (cv.contourArea(shape) > 500):
           
            (x , y , w , h) = cv.boundingRect(shape)

            cv.rectangle(first_frame , (x + w - 80, y + h - 40) , (x + w , y + h) , (0 , 0 , 0), 5)
    
    cv.imshow("shaps" , first_frame)

    cv.imshow("diffrent" , different)

    if cv.waitKey(1) & 0xFF == ord('o'):

        break

cv.destroyAllWindows()

camera.release()
