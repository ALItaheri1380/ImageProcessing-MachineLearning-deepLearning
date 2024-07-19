# computer vision, deep learning

## Deep learning

### Description
The subjects of these codes are:

1. Cancer Prediction using logistic regression
2. Read number from image by Cnn
3. Nationality prediction from a name by Rnn
4. Extract text from image
5. Student Grade Prediction by mlp
6. Image Classification
7. A very simple method to remove objects from photos
8. Text classification for comments about the hotel (by NLP)




## Cancer Prediction using logistic regression

> Using logistic regression, we find out whether a tumor is malignant or benign

![Annotation 2024-07-19 182442](https://github.com/user-attachments/assets/5993291c-c416-4225-bdae-e974f8357d35)

## Read number from image by Cnn

![Annotation 2024-07-19 182319](https://github.com/user-attachments/assets/9a5e8e86-1e40-49b4-b574-f910d4dcc2ae)

## Nationality prediction from a name by Rnn

![photo_2024-07-19_18-28-18](https://github.com/user-attachments/assets/48a2fd97-1c0a-40ad-8ed7-ba9dc225c6bd)


## Student Grade Prediction by mlp

![feduc-08-1106679-g005](https://github.com/user-attachments/assets/2ce3c80d-97fe-4d0b-ab7a-7328e1a496fc)


## image classification
We may want to recognize the nature of an animal or object, for example, to find out what kind of animal or object the photo we want is (dog or cat, etc.)


![photo1658265613](https://user-images.githubusercontent.com/98982133/179850660-1c54cdb5-15b8-414f-bcfe-b1f951fd5183.jpeg)


## Read text by pytesseract


![photo1658265897](https://user-images.githubusercontent.com/98982133/179851338-dfd68156-6ddf-4d38-8355-9e426c3007c7.jpeg)
![photo1658265912](https://user-images.githubusercontent.com/98982133/179851400-2e015c88-2396-4a6e-9aba-29eb41a419b1.jpeg)


## Remove objects from photos or videos or webcam

![remove](https://user-images.githubusercontent.com/98982133/183724316-c727b5d3-91a4-44b2-b5b5-f48ef6e105ea.png)

--------------------------------------------------------------------------------------------------------------------
## image processing


> Interesting processing on video and photos using Python

### Description
The subjects of these codes are:

1. to identify a person's face
2. recognize a person's face and their eyes and lips and the center of the face
3. identify hands and fingers and calculate the distance between two areas of the hand
4. identify a moving object

*Note that for some codes, you have to give them the desired link yourself

### All these pine examples are screenshots taken from a webcam and a webcam is needed to run each of the codes





## Detecting_Improper_Sitting

![photo_2023-08-16_02-03-44](https://github.com/ALItaheri1380/ImageProcessing-MachineLearning-deepLearning/assets/98982133/4b638adf-6110-49f6-a72c-c97e8010b1a3)

## Body Posture Detection


![BodyPostureDetection](https://user-images.githubusercontent.com/98982133/184558489-1dfe871c-be28-4161-88b2-132bdd0e5bcd.png)


## handgesture


![hand_gesture](https://user-images.githubusercontent.com/98982133/183730040-c9021f9d-8e31-4904-b013-58c46ca3df3a.png)

## colorize picture


![colorizepicture](https://user-images.githubusercontent.com/98982133/185639463-7ad5d466-05e0-459c-bc52-0d29e20cdcb2.png)


## faceDetector


![faceDetector](https://user-images.githubusercontent.com/98982133/183729247-6195bd3d-1fb9-4aa4-ba24-bf9f1f059094.png)



## EyeAndSmile


![EyeAndSmile](https://user-images.githubusercontent.com/98982133/183731323-a51bf7dc-9472-4e03-a66d-575ce6b98d1d.png)



## face_encoding

![face_encoding](https://user-images.githubusercontent.com/98982133/183731956-c8462a37-a61d-4aab-8b15-8d1e203e8d40.png)



## movingobjectDetector


![movingobjectDetector](https://user-images.githubusercontent.com/98982133/183732391-2cde7d81-0c3b-4b10-86bc-86a742f13fa0.png)



## template machin


![112](https://user-images.githubusercontent.com/98982133/179502509-3d94ad7c-61ee-4699-ad04-279810d1e753.png)



![12](https://user-images.githubusercontent.com/98982133/179502918-fe0304c0-38cc-4358-9be1-19bf12dc97dd.jpeg)

### plate_detection


Reading a text (in different languages) for example a car license plate


![h](https://user-images.githubusercontent.com/98982133/179608242-4b87dbfa-68f8-472e-95a1-2ef868f9159f.png)


## Computer vision assignment 

### HW1: Steganography

![Annotation 2024-07-19 211409](https://github.com/user-attachments/assets/7c088d9b-c282-4646-84d8-b35883808186)


### HW2: Image resizing

Methods used: 1. Bilinear, 2. Chessboard_Distance, 3. CityBlock_Distance, 4. Euclidean_Distance, 5. Neighbor, 6. My method(As can be seen in the report file, a significant increase in accuracy is observed)

![Annotation 2024-07-19 211611](https://github.com/user-attachments/assets/08348de4-aeb0-4131-a1ab-cca46bb816d3)

### HW3: part one: Image denoising(salt and pepper), part two: Removing noise from the photo and adding Persian numbers and displaying the result
> part one:

My innovative method is very accurate even against 90% noise and as can be seen in the report file, its psnr is much better than the famous methods.

![Annotation 2024-07-19 211115](https://github.com/user-attachments/assets/dfa62089-e9f7-41c7-aab3-d8fe58e78dfc)

> part two:

First, we remove the noises, find the range of numbers with the dfs algorithm, compare with the template photos and see which number it is, if it was red, it means positive, and if it was blue, it means negative (note that the photos are different sizes and we must use the resize algorithm) and place the result at the bottom of the photo (green number)

If you run the code, you will see that the accuracy of this method is 100% and there are no errors

![ResultImage_5_35](https://github.com/user-attachments/assets/56365a2c-e727-4acf-94be-a09c9c9fbc9c)


### HW4: Solve a puzzle

Solving a puzzle whose pieces are messed up, my method calculate the psnr of the edges and check which one has the most similarity and put the same piece of the puzzle in that place.

> Messed up image:

![Shuffled_Patches](https://github.com/user-attachments/assets/07bebbc2-e265-476f-83ab-f527247ec30b)

> Result

![photo_2024-07-19_21-00-56](https://github.com/user-attachments/assets/d6d46f1d-d7a5-4b56-a667-aac681740361)
