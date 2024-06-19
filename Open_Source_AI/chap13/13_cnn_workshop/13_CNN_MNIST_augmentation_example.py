# CNN MNIST with augmentation example

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import  numpy as np
import matplotlib.pyplot as plt

# define image size
img_rows=28
img_cols=28

# load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# reshape
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)


# one hot encoded
y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test)


# fix random seed for reproducibility 
seed = 100 
np.random.seed(seed)
num_classes = 10

# create CNN model
def cnn_model():
    # define model
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5), 
                            border_mode='valid',
                            strides=(1, 1),
                            input_shape=(img_rows, img_cols, 1), 
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(127, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


# build the model
model = cnn_model()

# Fit the model
disp = model.fit(X_train, y_train, 
          validation_data=(X_test, y_test), 
          nb_epoch=10, 
          batch_size=200, 
          verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("loss: %.2f" % scores[0])
print("acc: %.2f" % scores[1])