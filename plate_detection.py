import cv2 as cv

import easyocr

picture = cv.imread("Text.png" , 0)

# If the text is Persian

Read_Text = easyocr.Reader( ['fa'] )

# If the text is English

#Read_Text = easyocr.Reader( ['en'] )

Text = Read_Text.readtext(picture)

print(Text)
    