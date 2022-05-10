import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

import tkinter as tk
import tkinter.simpledialog
import tkinter.filedialog


# MNIST DATASET
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Classification
n_classes = 10
y_train_class = tf.keras.utils.to_categorical(y_train, n_classes)

# Normalize the train dataset
#x_train = tf.keras.utils.normalize(X_train, axis=1)

x_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
#x_test = X_test.reshape(10000, 28, 28, 1).astype('float32')



# Architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.summary()

#print(model.input_shape)
#print(model.weights)
#print(model.output_shape)

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
epochvalue = tk.simpledialog.askinteger("Input", "Enter total epoches for training")
model.fit(x_train, y_train_class, epochs=epochvalue)



tk.Tk().withdraw()
while 1:
    filename = tk.filedialog.askopenfile()
    if not filename:
        break
    img = cv2.imread(filename.name)[:,:,0]
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_LANCZOS4)
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    plt.title("Predicted Digit: {}".format(np.argmax(prediction)))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

