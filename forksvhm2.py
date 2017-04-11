
# coding: utf-8

# In[24]:

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.optimizers import SGD

from IPython.display import SVG,display
from keras.utils.vis_utils import model_to_dot

batch_size = 32
num_classes = 11
epochs = 10


# In[25]:

def hot(y):
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(11))
    new_y = np.zeros((y.shape[0],y.shape[1]*num_classes))
    for i in range(len(y)):
        new_y[i,:] = label_binarizer.transform(y[i]).flatten()
    return new_y


# In[26]:

X = np.load("modimages3.npy")
y = np.load("nummertrain.npy").astype(int)


# In[27]:

y = hot(y)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[40]:

y1,y2,y3,y4,y5,ynum = y_train[:,0:11],y_train[:,11:22],y_train[:,22:33],y_train[:,33:44],y_train[:,44:55],y_train[:,55:66]
y1_t,y2_t,y3_t,y4_t,y5_t,ynum_t = y_test[:,0:11],y_test[:,11:22],y_test[:,22:33],y_test[:,33:44],y_test[:,44:55],y_test[:,55:66]
print('x_train shape:', x_train.shape)
print('y_train shape:', ynum.shape)
print(x_train.shape[0], 'train samples')


# In[41]:

a = Input(shape=x_train.shape[1:])

x = Conv2D(32, (2, 2), padding='same')(a)
x = Activation('relu')(x)
x = Conv2D(32, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)


x = Conv2D(64, (4, 4), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (4, 4))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

conv_out = Flatten()(x)

num = Dense(128, activation='relu')(conv_out)
num = Dense(128, activation='relu')(num)
num = Dropout(0.5)(num)
numout = Dense(num_classes, activation='softmax',name="num")(num)

x1 = keras.layers.concatenate([conv_out, num])
x2 = keras.layers.concatenate([conv_out, num])
x3 = keras.layers.concatenate([conv_out, num])
x4 = keras.layers.concatenate([conv_out, num])
x5 = keras.layers.concatenate([conv_out, num])

x1 = Dense(128, activation='relu')(x1)
x2 = Dense(128, activation='relu')(x2)
x3 = Dense(128, activation='relu')(x3)
x4 = Dense(128, activation='relu')(x4)
x5 = Dense(128, activation='relu')(x5)


x1 = Dropout(0.5)(x1)
x2 = Dropout(0.5)(x2)
x3 = Dropout(0.5)(x3)
x4 = Dropout(0.5)(x4)
x5 = Dropout(0.5)(x5)


x1 = Dense(num_classes, activation='softmax', name='x1')(x1)
x2 = Dense(num_classes, activation='softmax', name='x2')(x2)
x3 = Dense(num_classes, activation='softmax', name='x3')(x3)
x4 = Dense(num_classes, activation='softmax', name='x4')(x4)
x5 = Dense(num_classes, activation='softmax', name='x5')(x5)

model = Model(inputs=a, outputs=[x1, x2, x3, x4, x5, numout])


display(SVG(model_to_dot(model).create(prog='dot', format='svg')))
# load weights
#model.load_weights("weights.forksvhmbest.hdf5")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[42]:

# checkpoint
filepath="weights.forksvhmbest.hdf5"
checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
callbacks_list = [checkpointer]


# In[43]:

model.fit(x_train, [y1,y2,y3,y4,y5,ynum],
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        shuffle=True,
        verbose=2,
        callbacks=[checkpointer])


# In[44]:

score = model.evaluate(x_test, [y1_t,y2_t,y3_t,y4_t,y5_t,ynum_t], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('forksvhm.h5')


# In[39]:

x_train.shape[1:]


# In[ ]:



