# -*- coding: utf-8 -*-
import csv
import cv2
import argparse
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Activation

import matplotlib.pyplot as plt

correction_factor = 0.2
image_path = ['data/IMG/','IMG/']
driving_log_path = ['data/driving_log.csv','driving_log.csv']
seperator = ['/','/']
epochs = 5
droprate = 0.2
modelpath='model_new1.h5'  

def load_driving_data():
    lines = []
    for i in range(1):
        with open(driving_log_path[i]) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                line.append(image_path[i])
                line.append(seperator[i])
                lines.append(line)
    return lines

def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def flipimg(image):
    return cv2.flip(image, 1)

def crop_and_resize(image):
    cropped = image[60:130, :]
    resized = cv2.resize(cropped, (160, 70))
    return resized

def process_batch(batch_sample):
    steering_angle = np.float32(batch_sample[3])
    images, steering_angles = [], []

    for image_path_index in range(3):
        image_name = batch_sample[image_path_index].split(batch_sample[8])[-1]
        image = cv2.imread(batch_sample[7] + image_name)
        rgb_image = bgr2rgb(image)
        resized = crop_and_resize(image)

        images.append(resized)

        if image_path_index == 1:
          steering_angles.append(steering_angle + correction_factor)
        elif image_path_index == 2:
          steering_angles.append(steering_angle - correction_factor)
        else:
          steering_angles.append(steering_angle)

        if image_path_index == 0:
          flipped_center_image = flipimg(resized)
          images.append(flipped_center_image)
          steering_angles.append(-steering_angle)

        return images, steering_angles

def data_generator(samples, batch_size=128):
    num_samples = len(samples)

    while True:
      shuffle(samples)

      for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset + batch_size]
        images, steering_angles = [], []

        for batch_sample in batch_samples:
          augmented_images, augmented_angles = process_batch(batch_sample)
          images.extend(augmented_images)
          steering_angles.extend(augmented_angles)

        X_train, y_train = np.array(images), np.array(steering_angles)
        yield shuffle(X_train, y_train)

def model():
    model = Sequential()
    model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)))
    
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2)))
    model.add(Activation('relu'))
#     model.add(Dropout(droprate))
    
    
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2)))
    model.add(Activation('relu'))
#     model.add(Dropout(droprate))
    
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2)))
    model.add(Activation('relu'))
#     model.add(Dropout(droprate))
    
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(Activation('relu'))
#     model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(Activation('relu'))
#     model.add(Dropout(droprate))
    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    
    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))
    
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


lines = load_driving_data();
train, validation = train_test_split(lines, test_size=0.2)

model = model()

history_object = model.fit_generator(generator=data_generator(train), validation_data=data_generator(validation),                         epochs= epochs,steps_per_epoch=len(train) * 2,validation_steps=len(validation),verbose=1)
model.save(modelpath)
print(history_object.history)

import pickle

# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([history_object.history], f)