from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

model = Sequential()

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                        input_shape=(64, 64, 3)
                       )
         )
model.add(MaxPooling2D(pool_size=(2, 2)))
def CRP(model, xCRP):
    for i in xCRP:
        model.add(Convolution2D(filters=32,
                                kernel_size=(3,3), 
                                activation='relu',
                               )
                 )
    return model

def Flat(model,xFlatten):
    model.add(Flatten())
    return model

def DenseLayer(model,xDense):
    model.add(Dense(units=128, activation='relu'))
    return xDense
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X,y)