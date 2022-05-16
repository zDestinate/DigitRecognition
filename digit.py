bdebug = 0
bAutoFindWrong = 0
bModifiedArchitecture = 1

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tkinter as tk
import tkinter.simpledialog
import tkinter.filedialog


# Load data from file
train_data = np.genfromtxt("train-data.txt", dtype=float, delimiter=" ")
X_train = train_data[:,1:]
Y_train = train_data[:,0].astype(int)

test_data = np.genfromtxt("test-data.txt", dtype=float, delimiter=" ")
X_test = test_data[:,1:]
Y_test = test_data[:,0].astype(int)



# Convert them into 16x16 image
x_train = X_train.reshape(7291, 16, 16, 1).astype('float32')
x_test = X_test.reshape(2007, 16, 16, 1).astype('float32')



# Classification
n_classes = 10
y_train_class = tf.keras.utils.to_categorical(Y_train, n_classes)
y_test_class = tf.keras.utils.to_categorical(Y_test, n_classes)


# Show training data image
if bdebug:
    plt.figure()
    for i in range(36):
        plt.subplot(6,6,i+1)
        plt.tight_layout()
        plt.imshow(x_train[i], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(Y_train[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()



# Architecture
model = tf.keras.models.Sequential()
if bModifiedArchitecture:
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(16, 16, 1), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
else:
    model.add(tf.keras.layers.Conv2D(6, input_shape=(16, 16, 1), kernel_size=(3, 3), activation="relu"))
    model.add(tf.keras.layers.AveragePooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu"))
    model.add(tf.keras.layers.AveragePooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(120, activation="relu"))
    model.add(tf.keras.layers.Dense(84, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
nBatchSize = tk.simpledialog.askinteger("Input", "Batch size\nMax: {0}".format(len(x_train)))
epochvalue = tk.simpledialog.askinteger("Input", "Enter total epoches for training")
epoch_model = model.fit(x_train, y_train_class, epochs=epochvalue, batch_size=nBatchSize, validation_data=(x_test, y_test_class))



# Epoch plotting
if bdebug:
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(epoch_model.history['accuracy'])
    plt.plot(epoch_model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(epoch_model.history['loss'])
    plt.plot(epoch_model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()
    plt.show(block=False)



# Test model eval
print("\n\nTest model evaluation:");
results = model.evaluate(x_test, y_test_class)
print("Test loss: {0}\nTest accuracy: {1}\nTest wrong: {2}\n\n".format(results[0], results[1], 1.0 - results[1]))



# Automatic find all the wrong
if bAutoFindWrong:
    total_wrong = 0
    for i in range(len(x_test)):
        prediction = model.predict(x_test[i].reshape(1, 16, 16))
        val_pred = np.argmax(prediction);
        val_truth = Y_test[i];
        if val_pred != val_truth:
            print("test[{0}] - \033[91mWRONG\033[00m".format(i))
            total_wrong += 1
    print("\nTotal Wrong: {0} out of {1}\n\n".format(total_wrong, len(x_test)))


# Manual test model
while 1:
    testmodel_index = tk.simpledialog.askinteger("Input", "Input the index # of the test model\nMax: {0}".format(len(x_test)))
    prediction = model.predict(x_test[testmodel_index].reshape(1, 16, 16))
    val_pred = np.argmax(prediction);
    val_truth = Y_test[testmodel_index];
    if val_pred == val_truth:
        output_test = '\033[92mCORRECT\033[00m'
    else:
        output_test = '\033[91mWRONG\033[00m'
    
    plt.figure()
    plt.title("Predicted: {0}\nTruth: {1}".format(val_pred, val_truth))
    plt.imshow(x_test[testmodel_index], cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    print("test[{0}] - Predicted: {1}, Truth: {2} {3}".format(testmodel_index, val_pred, val_truth, output_test))
    plt.show()


# Test file
# tk.Tk().withdraw()
# while 1:
    # filename = tk.filedialog.askopenfile()
    # if not filename:
        # break
    # img = cv2.imread(filename.name)[:,:,0]
    # img = cv2.resize(img, (16, 16), interpolation = cv2.INTER_LANCZOS4)
    # print(np.array([img]))
    # img = np.invert(np.array([img]))
    # prediction = model.predict(img)
    # plt.title("Predicted Digit: {}".format(np.argmax(prediction)))
    # plt.imshow(img[0], cmap=plt.cm.binary)
    # plt.show()

