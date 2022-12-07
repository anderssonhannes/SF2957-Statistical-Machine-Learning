import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import tensorflow as tf
# from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.images
y = digits.target

train_sample_size = 600
sample_size = round(train_sample_size/0.8)+100

test_s = 100/(sample_size)

X = X[0:sample_size]
y = y[0:sample_size]

# If you want full data set, uncomment following code
X = digits.images
y = digits.target
test_s = 100/len(X)

# Divide into test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=test_s, random_state=31)  

# Divide into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                        train_size=0.8, random_state=22) 

# Normalizing the data between 0-1 instead of 0-16
X_train = X_train[:,:,:,np.newaxis]/16.
X_val = X_val[:,:,:,np.newaxis]/16.
X_test = X_test[:,:,:,np.newaxis]/16.

y_train = y_train[:,np.newaxis]
y_val = y_val[:,np.newaxis]
y_test = y_test[:,np.newaxis]

plt.figure(figsize=(15,6))
for i in range(12):
    plt.subplot(2,6,i+1)
    plt.imshow(X_train[i].reshape((8,8)),cmap = "gray")
    plt.title(y_train[i,0])
    plt.axis('off')

# Setup for single layer
inputs = Input(shape = (8,8,1))

# Convolutional layer
conv_1 = Conv2D(filters = 8, padding='same',
                kernel_size = 3, strides = (1,1),activation = 'relu')(inputs)
mp_1 = MaxPool2D(pool_size= 2)(conv_1)

# Output layer
fl = Flatten()(mp_1)
outputs = Dense(10, activation = 'softmax')(fl)

model_single_layer = Model(inputs, outputs,name = "dense_single_layer")
opt = Adam()
model_single_layer.compile(optimizer = opt,
                        loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model_single_layer.summary()

training_hist = model_single_layer.fit(X_train, y_train,cbatch_size = 24, 
                                        epochs = 30, validation_data=(X_val,y_val))

y_pred = model_single_layer.evaluate(X_test,y_test,verbose=2)
print(round(training_hist.history['accuracy'][-1],3))
print(round(training_hist.history['val_accuracy'][-1],3))

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(training_hist.history['loss'])
plt.plot(training_hist.history['val_loss'])
plt.title('Loss')
plt.legend(['training','test'])
plt.subplot(2,1,2)
plt.plot(training_hist.history['accuracy'])
plt.plot(training_hist.history['val_accuracy'])
plt.title('accuracy')
plt.legend(['training','test'])
plt.show()

n_examples = 14
y_pred = model_single_layer.predict(X_val[0:n_examples])
plt.figure(figsize= (25,10))
for i in range(n_examples):
    plt.subplot(2,n_examples,i+1)
    plt.imshow(X_val[i].reshape((8,8)),cmap='gray')
    plt.title(y_val[i,0])
    plt.axis('off')
    plt.subplot(2,n_examples,i+n_examples+1)
    plt.bar(["0","1","2","3","4","5","6","7","8","9"],y_pred[i])
plt.show()

# Setup for double layer 
inputs = Input(shape = (8,8,1))

# Layer one
conv_1 = Conv2D(filters = 64, padding='same',kernel_size = 3, 
                strides = (1,1),activation = 'relu')(inputs)
mp_1 = MaxPool2D(pool_size= 2)(conv_1)

# Layer two
conv_2 = Conv2D(filters = 8, padding='same',kernel_size = 3,
                strides = (1,1),activation = 'relu')(mp_1)
mp_2 = MaxPool2D(pool_size= 2)(conv_2)

# Output layer
fl = Flatten()(mp_2)
outputs = Dense(10, activation = 'softmax')(fl)

model_double_layer = Model(inputs, outputs,name = "dense_double_layer")

opt = Adam()
model_double_layer.compile(optimizer = opt, 
                            loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model_double_layer.summary()

training_hist = model_double_layer.fit(X_train, y_train, batch_size = 24, 
                                        epochs = 30, validation_data=(X_val,y_val))

y_pred = model_double_layer.evaluate(X_test,y_test,verbose=2)
print(round(training_hist.history['accuracy'][-1],3))
print(round(training_hist.history['val_accuracy'][-1],3))

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(training_hist.history['loss'])
plt.plot(training_hist.history['val_loss'])
plt.title('Loss')
plt.legend(['training','test'])
plt.subplot(2,1,2)
plt.plot(training_hist.history['accuracy'])
plt.plot(training_hist.history['val_accuracy'])
plt.title('accuracy')
plt.legend(['training','test'])
plt.show()

n_examples = 14
y_pred = model_double_layer.predict(X_val[0:n_examples])
plt.figure(figsize= (25,10))
for i in range(n_examples):
    plt.subplot(2,n_examples,i+1)
    plt.imshow(X_val[i].reshape((8,8)),cmap='gray')
    plt.title(y_val[i,0])
    plt.axis('off')
    plt.subplot(2,n_examples,i+n_examples+1)
    plt.bar(["0","1","2","3","4","5","6","7","8","9"],y_pred[i])
plt.show()
