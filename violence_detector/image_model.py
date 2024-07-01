import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import os
from PIL import Image
from io import BytesIO
import time

"""
We get pretrained weight and refer to the article: 
Robust Real-Time Violence Detection in Video Using CNN And LSTM

"""

class ViolenceDetector:
    def __init__(self, config):
        self.pretrained_weight_path = config["pretrained_weight_path"]
        self.layers = tf.keras.layers
        self.models = tf.keras.models
        self.losses = tf.keras.losses
        self.optimizers = tf.keras.optimizers
        self.metrics = tf.keras.metrics
        self.num_classes = 2
        
        self.input_shapes=(160, 160, 3)
        self.vg19 = tf.keras.applications.vgg19.VGG19

        self.num_frames_to_predict = config["num_frames_to_predict"]

    def get_model(self):
        """
        Get the model with the weight loaded from pretrained weight
        """
        print("Loading model for violence detector...")
        np.random.seed(1234)

        base_model = self.vg19(include_top=False,weights='imagenet',input_shape=(160, 160,3))
        cnn = self.models.Sequential()
        cnn = self.models.Sequential()
        cnn.add(base_model)
        cnn.add(self.layers.Flatten())
        final_model = self.models.Sequential()
        final_model.add(self.layers.TimeDistributed(cnn, input_shape=(self.num_frames_to_predict, 160, 160, 3)))
        final_model.add(self.layers.LSTM(30 , return_sequences= True))

        final_model.add(self.layers.TimeDistributed(self.layers.Dense(90)))
        final_model.add(self.layers.Dropout(0.1))

        final_model.add(self.layers.GlobalAveragePooling1D())

        final_model.add(self.layers.Dense(512, activation='relu'))
        final_model.add(self.layers.Dropout(0.3))

        final_model.add(self.layers.Dense(self.num_classes, activation="sigmoid"))

        adam = self.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        final_model.load_weights(self.pretrained_weight_path)
        rms = self.optimizers.RMSprop()

        final_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

        return final_model
