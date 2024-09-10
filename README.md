# computer vision, deep learning

## Deep learning

### Description
The subjects of these codes are:

1. VAE for MNIST dataset
2. Cancer Prediction using logistic regression
3. Read number from image by Cnn
4. Nationality prediction from a name by Rnn
5. Extract text from image
6. Student Grade Prediction by mlp
7. Image Classification
8. A very simple method to remove objects from photos
9. Text classification for comments about the hotel (by NLP)


# VAE for mnist dataset

![Annotation 2024-09-10 140639](https://github.com/user-attachments/assets/f1e9b659-b5e0-476c-810a-50431ebdaf43)


A **Variational Autoencoder (VAE)** is a type of generative model that can learn to encode data (like images) into a compressed latent space and then decode it back to reconstruct the original input. VAEs are particularly useful for generating new data samples similar to the input data. Let's discuss the concept and how a VAE can be applied to the MNIST dataset.

### Overview of VAE:

1. **Encoder**: This part of the model takes an input (e.g., an MNIST digit image) and maps it to a distribution in the latent space (usually a normal distribution). Instead of encoding the input directly as a single point in the latent space, the encoder outputs parameters of a distribution (mean and variance).

2. **Latent Space**: This is a lower-dimensional space where the data is represented. A VAE learns to represent the input data distribution in this space, making it easier to sample new data points.

3. **Decoder**: The decoder maps points from the latent space back to the original data space. Given a point in the latent space, the decoder tries to reconstruct an image (e.g., an MNIST digit).

4. **Loss Function**: The VAE uses a combination of two losses:
   - **Reconstruction Loss**: Measures how well the reconstructed output matches the input (e.g., pixel-wise difference between original and generated images).
   - **KL Divergence Loss**: Measures how close the learned latent space distribution is to a prior distribution (typically a standard normal distribution).

### Applying VAE to MNIST

#### Steps:

1. **Load the MNIST dataset**:
   MNIST is a dataset of handwritten digits. Each image is 28x28 pixels, grayscale, and labeled with the corresponding digit (0-9).

2. **Define the Encoder**:
   The encoder takes the input image and outputs two vectors, representing the mean and log variance of the latent distribution.

3. **Latent Sampling**:
   Use the mean and variance to sample from the latent space using the "reparameterization trick," which allows gradients to flow during backpropagation.

4. **Define the Decoder**:
   The decoder takes samples from the latent space and tries to reconstruct the original image.

5. **Train the VAE**:
   Train the model using both reconstruction loss and KL divergence to ensure that the latent space is organized and can generate valid images.
   
-----
# GAN for MNIST and CIFAR-10 dataset

![Annotation 2024-09-10 141817](https://github.com/user-attachments/assets/1267069a-7934-47a7-9a7b-0f58a8816d25)


A **Generative Adversarial Network (GAN)** is a type of neural network architecture used to generate new data points from an existing dataset. GANs consist of two components:

1. **Generator**: Generates fake samples from random noise.
2. **Discriminator**: Distinguishes between real and fake samples.

These two networks are trained simultaneously: the generator tries to fool the discriminator by generating realistic data, while the discriminator tries to correctly identify real vs. fake data. Over time, the generator gets better at creating realistic data.

Let's walk through how to implement a GAN for both the **MNIST** (handwritten digits) and **CIFAR-10** (natural images) datasets using TensorFlow/Keras.

### Steps:
1. **Load the Dataset**: For both MNIST and CIFAR-10.
2. **Build the Generator**: A neural network that generates fake images from random noise.
3. **Build the Discriminator**: A neural network that classifies images as real or fake.
4. **GAN Model**: The combination of generator and discriminator.
5. **Training Loop**: Train the generator and discriminator iteratively.
   
------

# Cancer Prediction using logistic regression

![Annotation 2024-07-19 182442](https://github.com/user-attachments/assets/5993291c-c416-4225-bdae-e974f8357d35)


## Overview

This project implements a logistic regression model to predict cancer diagnoses based on patient data. The data is read from a CSV file and includes various patient attributes related to cancer.

## Dataset

The dataset used for training and testing the model includes the following columns:

- **S/N**: Serial number
- **Year**: Year of data collection
- **Age**: Age of the patient
- **Menopause**: Menopause status (1 for post-menopausal, 0 for pre-menopausal)
- **Tumor Size (cm)**: Size of the tumor in centimeters
- **Inv-Nodes**: Number of lymph nodes with cancer
- **Breast**: Side of the breast (Left or Right)
- **Metastasis**: Presence of metastasis (1 for yes, 0 for no)
- **Breast Quadrant**: Quadrant of the breast where the tumor is located
- **History**: Family history of cancer (1 for yes, 0 for no)
- **Diagnosis Result**: Result of the diagnosis (Benign or Malignant)

## Instructions

1. **Data Preparation**: Ensure that the dataset is in CSV format and properly formatted with the columns as described above.

2. **Loading the Data**: Use a library like `pandas` to read the CSV file into a DataFrame.

3. **Feature Selection**: Choose relevant features for the logistic regression model and preprocess the data as necessary (e.g., encoding categorical variables).

4. **Model Training**: Split the data into training and testing sets. Train the logistic regression model using the training set.

5. **Evaluation**: Evaluate the model's performance using metrics such as accuracy, precision, recall, and the ROC curve.

6. **Prediction**: Use the trained model to make predictions on new data.

-------

# Number Reading from Images Using Convolutional Neural Networks (CNN)

![Annotation 2024-07-19 182319](https://github.com/user-attachments/assets/9a5e8e86-1e40-49b4-b574-f910d4dcc2ae)

## Overview

This project utilizes a Convolutional Neural Network (CNN) to read and classify numbers and symbols from images. The dataset contains images labeled with various classes, including digits and special characters.

## Dataset

The dataset consists of 60,000 images with the following classes:

- **Digits**: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
- **Special Characters**: 'dot', 'minus', 'plus', 'slash', 'x', 'y', 'z', 'w'

Each image in the dataset is labeled according to one of these classes.

## Instructions

1. **Data Preparation**: Ensure the dataset of 60,000 images is organized and labeled correctly. The images should be accessible in a format suitable for loading into the CNN model.

2. **Loading the Data**: Use `PyTorch` library  to load and preprocess the images. Normalize the images and perform any necessary data augmentation.

3. **Model Architecture**: Define the architecture of the CNN. Typical layers include convolutional layers, activation functions, pooling layers, and fully connected layers.

4. **Training the Model**: Split the dataset into training and validation sets. Train the CNN model using the training set, monitoring its performance on the validation set.

5. **Evaluation**: Evaluate the model's performance using metrics such as accuracy and loss. Fine-tune the model parameters as needed.

6. **Prediction**: Use the trained model to predict classes for new images.

----------------

# Nationality Prediction from Names Using Recurrent Neural Networks (RNN)

![photo_2024-07-19_18-28-18](https://github.com/user-attachments/assets/48a2fd97-1c0a-40ad-8ed7-ba9dc225c6bd)

## Overview

This project uses a Recurrent Neural Network (RNN) to predict the nationality of a person based on their name. The RNN model is trained to classify names into different nationalities by learning patterns in character sequences.

## Dataset

The dataset consists of names labeled with their corresponding nationalities. Each name is used to train the RNN to recognize patterns that are indicative of different nationalities.

## Instructions

1. **Data Preparation**: Ensure the dataset is structured with names and their associated nationalities. The dataset should be in a format compatible with the provided code, typically CSV or similar.

2. **Loading the Data**: Use the provided Jupyter Notebook to load and preprocess the dataset. This involves encoding names and nationalities into formats suitable for the RNN model.

3. **Model Architecture**: The notebook defines an RNN model that includes layers such as embedding, LSTM/GRU, and dense layers. Review and adjust the architecture as needed for your dataset.

4. **Training the Model**: Split the data into training and validation sets. Train the RNN model using the training set, monitoring its performance on the validation set.

5. **Evaluation**: Evaluate the model's performance using accuracy and other relevant metrics. Adjust model parameters and hyperparameters to improve performance.

6. **Prediction**: Use the trained RNN model to predict the nationality of new names.

----------

# Student Grade Prediction Using Multi-Layer Perceptron (MLP)

![feduc-08-1106679-g005](https://github.com/user-attachments/assets/2ce3c80d-97fe-4d0b-ab7a-7328e1a496fc)


## Overview

This project uses a Multi-Layer Perceptron (MLP) to predict student grades based on various input features. The MLP model is trained to make predictions on student performance by learning from historical data.

## Dataset

The dataset contains features related to student performance and the target variable is the student's grade. Each record includes input features such as study hours, sleep hours
, and other relevant attributes.

## Instructions

1. **Data Preparation**: Ensure that the dataset is organized with relevant features and the target variable (grades). The dataset should be in a format compatible with the provided code, typically CSV or similar.

2. **Loading the Data**: Use the provided Jupyter Notebook to load and preprocess the dataset. This includes handling missing values, encoding categorical variables, and splitting the data into training and test sets.

3. **Model Architecture**: The notebook defines an MLP model with one or more hidden layers. Review the architecture and adjust the number of layers, neurons, activation functions, and other parameters as needed.

4. **Training the Model**: Train the MLP model using the training dataset. Monitor performance metrics such as accuracy or mean squared error on the validation set.

5. **Evaluation**: Evaluate the model's performance on the test set using metrics such as accuracy, mean squared error, or R-squared. Fine-tune the model parameters as necessary to improve performance.

6. **Prediction**: Use the trained MLP model to make predictions on new student data.

-------

## image classification

![photo1658265613](https://user-images.githubusercontent.com/98982133/179850660-1c54cdb5-15b8-414f-bcfe-b1f951fd5183.jpeg)

## Overview

This project focuses on image classification using TensorFlow. The goal is to recognize and classify different types of animals or objects from images. For example, the model can identify whether an image contains a dog, cat, or other specified categories.

## Dataset

The dataset consists of labeled images of various animals or objects(CIFAR-10). Each image is tagged with its corresponding class label, such as 'dog', 'cat', or other categories. Ensure that the dataset is well-organized and split into training and test sets.

## Instructions

1. **Data Preparation**: Organize and preprocess the dataset. This may involve resizing images, normalizing pixel values, and augmenting the data to improve model performance. Ensure that images are labeled correctly and stored in a structured format.

2. **Loading the Data**: Use TensorFlow and `tf.data` API to load and preprocess the images. This involves creating a dataset pipeline that efficiently handles image loading, preprocessing, and batching.

3. **Model Architecture**: Define the architecture of the convolutional neural network (CNN) using TensorFlow. Typical architectures include layers such as convolutional layers, pooling layers, and fully connected layers.

4. **Training the Model**: Train the CNN model on the training dataset. Use techniques such as dropout and regularization to prevent overfitting. Monitor the model's performance on the validation set and adjust hyperparameters as necessary.

5. **Evaluation**: Evaluate the model's accuracy and performance on the test set using metrics such as accuracy, precision, recall, and F1 score. Analyze the results to ensure the model meets the desired performance criteria.

6. **Prediction**: Use the trained model to classify new images. Implement a function to preprocess the input image and obtain predictions from the model.

----------------------

## Read text by pytesseract
![photo1658265897](https://user-images.githubusercontent.com/98982133/179851338-dfd68156-6ddf-4d38-8355-9e426c3007c7.jpeg)
![photo1658265912](https://user-images.githubusercontent.com/98982133/179851400-2e015c88-2396-4a6e-9aba-29eb41a419b1.jpeg)

This project uses PyTesseract, an OCR (Optical Character Recognition) tool, to extract text from images. PyTesseract is a Python wrapper for Google's Tesseract-OCR Engine.

-------

### Remove objects from photos or videos or webcam

![remove](https://user-images.githubusercontent.com/98982133/183724316-c727b5d3-91a4-44b2-b5b5-f48ef6e105ea.png)

--------------------------------------------------------------------------------------------------------------------
# image processing
## Interesting processing on video and photos using Python

### Description
The subjects of these codes are:

1. to identify a person's face
2. recognize a person's face and their eyes and lips and the center of the face
3. identify hands and fingers and calculate the distance between two areas of the hand
4. identify a moving object


# Detecting Improper Sitting with Computer Vision

![photo_2023-08-16_02-03-44](https://github.com/ALItaheri1380/ImageProcessing-MachineLearning-deepLearning/assets/98982133/4b638adf-6110-49f6-a72c-c97e8010b1a3)


## Overview

This project uses computer vision techniques to detect improper sitting posture in real-time. By analyzing video input, the system assesses body alignment and posture, providing feedback on whether the sitting posture is correct or incorrect.

## Code Overview

The code performs the following tasks:

1. **Setup**: Utilizes OpenCV for video capture, Mediapipe for pose estimation, and Win32 API for sound alerts.

2. **Posture Detection**:
   - **Find Distance**: Calculates the distance between shoulder points to determine alignment.
   - **Find Angle**: Measures neck and torso inclinations to evaluate posture.

3. **Real-Time Analysis**:
   - Captures video frames and processes them to extract key points for posture assessment.
   - Determines if the posture is correct based on predefined angle thresholds.
   - Displays real-time feedback on the screen, including posture metrics and alerts.

4. **Alerts**:
   - Sounds an alert if bad posture is detected for more than 5 seconds.

## Instructions

1. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe pywin32
   ```

2. **Run the Script**: Execute the script to start video capture and posture analysis.
   ```bash
   python detect_improper_sitting.py
   ```

3. **Usage**: Adjust posture in front of the camera. The system will provide feedback on your sitting posture and sound an alert if necessary.

-------------------

# Body Posture Detection with OpenCV

![BodyPostureDetection](https://user-images.githubusercontent.com/98982133/184558489-1dfe871c-be28-4161-88b2-132bdd0e5bcd.png)


## Overview

This project employs OpenCV to detect and analyze body posture in real-time. The system captures video input and evaluates body alignment to determine if the posture is correct or incorrect.

## Key Features

- **Real-Time Detection**: Analyzes video frames to assess body posture.
- **Pose Estimation**: Uses OpenCV's capabilities to identify key body landmarks.
- **Posture Feedback**: Provides visual feedback on posture alignment.

## Instructions

1. **Install Dependencies**:
   ```bash
   pip install opencv-python
   ```

2. **Run the Script**: Start the posture detection system by running the provided script.
   ```bash
   python posture_detection.py
   ```

3. **Usage**: Position yourself in front of the camera. The system will display real-time feedback on your posture and alert you if your posture deviates from the expected alignment.

----------

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

# seam carving for content-aware image resizing

**Seam carving** is a technique used for content-aware image resizing. Instead of uniformly shrinking or expanding an image, seam carving intelligently removes or adds pixels along paths of least importance, allowing important features (like people or objects) to remain undistorted.

### Key Concepts:

1. **Seams**: A seam is a connected path of pixels that extends from one side of the image to the opposite side (top to bottom or left to right). The seam is chosen to have the least significance (based on some energy function).
   
2. **Energy Function**: This function calculates the "importance" of a pixel. Common choices include the gradient magnitude of the image (e.g., using Sobel filters) to find areas of high contrast or edge information, which are considered more important.

3. **Seam Removal**: The process involves finding the seam with the lowest energy and removing it, reducing the image size without distorting important content.

4. **Seam Insertion**: To enlarge an image, seams with the lowest energy can be duplicated, which adds pixels to the image without distorting key features.

### Steps of Seam Carving:
1. **Compute Energy Map**: 
   - An energy map is created by calculating the gradient magnitude of the image. This highlights areas of high importance (like edges).
   
2. **Find Optimal Seam**: 
   - Using dynamic programming, the seam with the lowest cumulative energy is found. This seam is a continuous path of connected pixels that spans the image.

3. **Remove or Insert Seam**: 
   - For reducing size, remove the seam. For enlarging, duplicate the seam.

![castle1-1](https://github.com/user-attachments/assets/171458be-2c41-4211-8552-7f5f763b7cb9)


# Computer vision assignment 

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

![photo_2024-07-19_22-04-45](https://github.com/user-attachments/assets/3183448c-162d-4c09-8157-e7fae0487578)


> Result

![photo_2024-07-19_21-00-56](https://github.com/user-attachments/assets/d6d46f1d-d7a5-4b56-a667-aac681740361)
