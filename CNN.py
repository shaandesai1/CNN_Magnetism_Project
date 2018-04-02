
# coding: utf-8

# In[27]:

# import libraries
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Conv3D,MaxPooling3D
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model



# #### Core dataframe processing functions

# In[28]:

def make_target(y_train):
    """ create ytarget var from magnetic moment values """
    print(np.mean(y_train))
    low_dex = np.where(y_train < 4)[0]
    hi_dex = np.where(y_train >= 4)[0]
    y_train[low_dex] = 0
    y_train[hi_dex] = 1
    return y_train

def resize_data(x_train):
    """
        set all input charge densities to have same dimension.. Trim down to z=320
    """
    N = x_train.shape[0]
    zsize = 320
    x_train_M = np.empty((N,60,60,zsize))
    for ith, xtr in enumerate(x_train):
        x_train_M[ith,:,:,:] = xtr[0,:,:,:zsize]
    return x_train_M

def train_test(df_charge,train_size):
    """ 
        Get training and test data 
        * Train_size input is fraction of total data size
    """   
    N = len(df_charge)
    #print(N)
    randex = np.random.permutation(np.arange(N))
    trN = np.int(np.floor(train_size*N))
    print(trN)
    charge_data = df_charge['sorted_charge_data'].values
    mag_data = df_charge['mag_mom'].values
    x_train = charge_data[:trN].copy()
    y_train = mag_data[:trN].copy()
    x_test = charge_data[trN:].copy()
    y_test = mag_data[trN:].copy()
    #resize:
    print(x_train.shape)
    x_train = resize_data(x_train)
    print(x_train.shape)
    x_test = resize_data(x_test)
    return x_train, y_train, x_test, y_test

#create channels for the image
#only run this once, otherwise you'll keep appending new values at the end
def channels(x_t):
    """
    create channels for the image - currently creates 2 channels (positive and negative)
    only run this once, otherwise you'll keep appending new values at the end
    """
    xt1 = np.copy(x_t)
    xt2 = np.copy(x_t)
    xt1[xt1<0] = 0
    xt2[xt2>0] = 0
    old_shape = list(xt1.shape)
    old_shape.append(2)
    new_shape = tuple(old_shape)
    newvec = np.zeros(new_shape)
    newvec[:,:,:,:,0] = xt1
    newvec[:,:,:,:,1] = xt2
    return newvec
    


# In[29]:

#read input from splitter
df_charge=pd.read_pickle('chgdf_input')


# In[30]:

#train test split
train_size = 0.70
x_train, y_train, x_test, y_test = train_test(df_charge,train_size)


# In[31]:

# input image dimensions
img_rows, img_cols = 60, 60


# In[32]:

trainsize = x_train.shape[0]
testsize = x_test.shape[0]


# In[33]:


# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train, x_test, y_test = train_test(df_charge, train_size)
print('x_train.shape', x_train.shape)
## X_train is of shape n_samples x 28 x 28
## for a CNN we want to keep the image shape
## need to explicitly tell keras that it is a gray value image
## so each image is 60x60x1 not 28x28x3

#take a slice out of the volume for training and analysis
slice = 120
x_train = x_train[:,:,:,:slice]
x_test = x_test[:,:,:,:slice]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, slice)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, slice)
input_shape = (img_rows, img_cols, slice)

# normalize image values to [0,1]
# interestingly the keras example code does not center the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#do not think this applies - not using images
# x_train /= 255
# x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[34]:


#replicate the charge matrix 
#dimx and dimy define the scale at which you want to replicate thus 1,1 means return input
#dimx and dimy need to be integers!
def matmul(mat,dimx,dimy):
    """
    replicate the charge matrix in the x and y directions by scaling factor of dimx,dimy
    dimx,dimy need to be integers
    """
    xrep = np.shape(mat)[0]
    yrep = np.shape(mat)[1]
    zrep = np.shape(mat)[2]
    ret_mat = np.zeros((dimx*xrep,dimy*yrep,1*zrep))
    #iterate for the integer multiple
    for i in range(int(dimx)):
        for j in range(int(dimy)):
            ret_mat[i*xrep:(i+1)*xrep,yrep*j:(j+1)*yrep,:] = mat
    return ret_mat


new_xtrain = []
new_xtest = []
for i in range(x_train.shape[0]):
    new_xtrain.append(matmul(x_train[i,:,:,:],2,2))
for i in range(x_test.shape[0]):
    new_xtest.append(matmul(x_test[i,:,:,:],2,2))
x_train = np.array(new_xtrain)
x_test = np.array(new_xtest)

#print(np.shape(x_train))


# In[35]:

#print(np.shape(x_test))


# In[36]:

#call channels which splits this into a positive and negative sparse matrix
num_classes = 2

x_test = channels(x_test)
x_train = channels(x_train)
print('x_test.shape, y_test.shape, x_train.shape, y_train.shape')
print(x_test.shape, y_test.shape, x_train.shape, y_train.shape)

y_train = make_target(y_train)
y_test = make_target(y_test)

# convert class vectors to binary class matrices
# keras likes one hot encoding instead of class names
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)
print(y_train.shape)


# In[38]:

# create an empty network model
model = Sequential()

# --- input layer ---
#no padding #width,height,depth,channels
model.add(Conv3D(3, kernel_size=(40,40,40), activation='relu', input_shape=(120,120,120,2)))
# --- max pool ---
model.add(MaxPooling3D(pool_size=(10,10,10)))

# --- next layer ---
# we could double the number of filters as max pool made the 
# feature maps much smaller 
# just not doing this to improve runtime
model.add(Conv3D(5, kernel_size=(3,3,3), activation='relu'))
# --- max pool ---
model.add(MaxPooling3D(pool_size=(2,2,2)))

# flatten for fully connected classification layer
model.add(Flatten())
# note that the 2 is the number of classes we have
# the classes are mutually exclusive so softmax is a good choice
# --- fully connected layer ---
model.add(Dense(12, activation='relu'))

# --- classification ---
model.add(Dense(2, activation='softmax'))

# prints out a summary of the model architecture
#model.summary()


# In[39]:

# this does all necessary compiling. In tensorflow this is much quicker than in theano
# the setup is our basic categorical crossentropy with stochastic gradient decent
# we also specify that we want to evaluate our model in terms of accuracy
sgd = SGD(lr=1, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# model.compile(loss='mean_squared_error',
#               optimizer='adam')


# In[ ]:

# this is now the actual training
# in addition to the training data we provide validation data
# this data is used to calculate the performance of the model over all the epochs
# this is useful to determine when training should stop
# in our case we just use it to monitor the evolution of the model over the training epochs
# if we use the validation data to determine when to stop the training or which model to save, we 
# should not use the test data, but a separate validation set. 
batch_size = 80

model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=1,
            verbose=1,
            validation_data=(x_test, y_test))

# # once training is complete, let's see how well we have done
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# In[ ]:

model.save('cnn_test1.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

