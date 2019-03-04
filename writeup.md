# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeImage/OriginalImage.png "Original Image"
[image2]: ./writeImage/org_bgr2rgb.png "BRG to RGB converted"
[image3]: ./writeImage/org_cropped_and_resized.png "Cropped and resized"
[image4]: ./writeImage/flipedImage.png "Flipped Image"
[image5]: ./writeImage/flip_bgr2rgb.png "Flipped BRG to RGB converted"
[image6]: ./writeImage/flipped_cropped_resized.png "Flipped Cropped and resized"
[image7]: ./writeImage/modelLayer.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behaviorial_cloning.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 Autonomyous Driving video by the model 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The behaviorial_cloning.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64. The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

To address overfitting in tried below options.
* Used left and Right camera image. Used Correction for stearing -.25 and .25 respectively.
* Fliped the center camera image and added to the training set to get more data.

#### 3. Model parameter tuning

Batch Size = 124
Optimizer = adam
loss = MSE
EPOCHS = 5

#### 4. Appropriate training data
 
I tried to capture the Training data by driving the Car Stimulater provide as part of the  Lab.
I captured data by driving the car in Normal direction of the lap and also by drove the car in reverse direction so that trained model doesn't  align on same direction.

Also included the data provided as part of the Lab in training set.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used the NVIDIA architecture as suggested by the lecture and created the model.

I trained the model on the dataset captured from the stimulator.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Since I was using data augmentation techniques, the mean squared error was low both on the training and validation steps.

#### 2. Final Model Architecture

The final model architecture was pretty much same as the staring approcch. I have used below Image processing steps.

My Model has Below layers:

![alt text][image7]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded multiple laps on track with diffrent combination like driving in center for one lap and movind from left to right and right to left so that model can predict the steering angles in tough sections of the lane.

Also while preparing the data in generator i have fliped the Center Camera Image so as to get more data.

I  augmentented the data set as below:
 * Converted the Image to BGR2RGB
 * Cropped the top part of the Image
 * Resized the Image to size [160, 70]

## Sample Augumented Images:
### Original Center Camera Image
![alt text][image1]
### BGR2RGB Converted
![alt text][image2]
### Croped and Resized Image
![alt text][image3]
### Fliped Center Camera Image
![alt text][image4]
### Fliped BGR2RGB Converted 
![alt text][image5]
### Fliped Croped and Resized Image
![alt text][image6]



