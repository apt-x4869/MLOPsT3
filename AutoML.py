# ### Let's construct LeNet in Keras! 
# #### First let's load and prep our MNIST data

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras
import pandas as pd

# loads the MNIST dataset
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# Lets store the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# Getting our date in the right 'shape' needed for Keras
# We need to add a 4th dimenion to our date thereby changing our
# Our original image shape of (60000,28,28) to (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# store the shape of a single image 
input_shape = (img_rows, img_cols, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

# ### Now let's create our layers to replicate LeNet
# create model
model = Sequential()

# CRP , Flatten , Dense Layer functions
def CRP(model, xCRP, input_shape=None, pool_size=(2,2)):
    for i in range(1,xCRP+1):
        if i ==1 and input_shape != None :
            model.add(Conv2D(filters=20, kernel_size=(3,3), padding="same", input_shape=input_shape))
            model.add(Activation("relu"))
            continue
        model.add(Conv2D(filters=20, kernel_size=(3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=pool_size))
    return model
def Flat(model):
    model.add(Flatten())
    return model
def DenseLayer(model,xDense,output_shape=128, units=500):
    for i in range(1,xDense+1) :
        if i == (xDense) and output_shape != None:
            model.add(Dense(units=output_shape))
            model.add(Activation("softmax"))
            continue
        model.add(Dense(units=units))
        model.add(Activation("relu"))
    return model

def modelCompile(model, no_CRP_layers, no_Dense_layers, pool_size=None, input_shape=None,output_shape=None):
    model = CRP(model, no_CRP_layers, input_shape=input_shape, pool_size=pool_size)
    model = Flat(model)
    model = DenseLayer(model, no_Dense_layers, num_classes)
    model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])
    return model

# Model HyperParameters
a = pd.read_csv('hPara.csv')
a = a.where(pd.notnull(a), None)
flag=0
try :
    no_CRP_layers = int(a.iloc[-1,0])
except:
    no_CRP_layers = 1
    flag=1
try :
    no_Dense_layers = int(a.iloc[-1,1]) 
except:
    no_Dense_layers = 1
pool_size=(2,2)

# Training Parameters
try:
    epochs=int(a.iloc[-1,2])
except:
    epochs = 1
batch_size = 128

# 2 sets of CRP (Convolution, RELU, Pooling)
# Fully connected layers (w/ RELU)          
# Softmax (for classification)
model = modelCompile(model, no_CRP_layers, no_Dense_layers, pool_size=(2,2),input_shape=input_shape, output_shape=num_classes)

print(model.summary())

# Now Let's Train our model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, verbose=0)

#Saving Model
model.save("mnist_LeNet.h5")

# Evaluate the performance of our trained model
scores = model.evaluate(x_test, y_test, verbose=0)
if flag==1:
    x = { 'no_CRP_layers':no_CRP_layers,
         'no_Dense_layers':no_Dense_layers,
         'epochs':epochs,
         'acc':scores[1]
        }
    print(x)
    a = a.append(x,ignore_index=True,sort=False)
else:
    a.iloc[-1,0]=no_CRP_layers
    a.iloc[-1,1]=no_Dense_layers
    a.iloc[-1,2]=epochs
    a.iloc[-1,3]=scores[1]
a.to_csv('hPara.csv',index=False) 
print(a)

