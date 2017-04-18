
# coding: utf-8

# In[10]:

import numpy as np
#from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.optimizers import SGD

from IPython.display import SVG,display
from keras.utils.vis_utils import model_to_dot

batch_size = 32
num_classes = 11
epochs = 15


# In[6]:

def hot(y):
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(11))
    new_y = np.zeros((y.shape[0],y.shape[1]*num_classes))
    for i in range(len(y)):
        new_y[i,:] = label_binarizer.transform(y[i]).flatten()
    return new_y


# In[7]:

X = np.load("resizecrop.npy")
Y = np.load("nummertrain.npy").astype(int)


# In[8]:

y = hot(Y)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)


# In[9]:

y1,y2,y3,y4,y5,ynum = y_train[:,0:11],y_train[:,11:22],y_train[:,22:33],y_train[:,33:44],y_train[:,44:55],y_train[:,55:66]
y1_t,y2_t,y3_t,y4_t,y5_t,ynum_t = y_test[:,0:11],y_test[:,11:22],y_test[:,22:33],y_test[:,33:44],y_test[:,44:55],y_test[:,55:66]
print('x_train shape:', x_train.shape)
print('y_train shape:', ynum.shape)
print(x_train.shape[0], 'train samples')


# In[11]:

inpu = Input(shape=x_train.shape[1:])

x = Conv2D(32, (2, 2), padding='same')(inpu)
x = Activation('relu')(x)
x = Conv2D(32, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (4, 4), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (4, 4))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

conv_out = Flatten()(x)

num = Dense(128, activation='relu')(conv_out)
num = Dense(128, activation='relu')(num)
num = Dropout(0.5)(num)
numout = Dense(num_classes, activation='softmax',name="num")(num)

numtower = Model(inputs=inpu, outputs=numout)


# display(SVG(model_to_dot(model).create(prog='dot', format='svg')))
# load weights
# model.load_weights("weights.forksvhmbest.hdf5")

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]

numtower.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[12]:

numtower.fit(x_train, ynum,
        batch_size=batch_size,
        epochs=epochs,
        #validation_split=0.2,
        validation_data=(x_test,ynum_t),
        shuffle=True,
        verbose=2,
        callbacks=callbacks)

score = numtower.evaluate(x_test, ynum_t, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

numtower.save('ioncanonnum.h5')

print()
print("Number tower complete")
print()

# In[ ]:

numtowerfre = load_model('ioncanonnum.h5')
numtowerfre.name ="Numbertower"
print("starting Digit1")

inpu = Input(shape=x_train.shape[1:])

# Number tower
numtower = numtowerfre(inpu)
numtower.trainable=False

x = Conv2D(32, (2, 2), padding='same')(inpu)
x = Activation('relu')(x)
x = Conv2D(32, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (4, 4), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (4, 4))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

conv_out = Flatten()(x)


x1 = keras.layers.concatenate([conv_out, numtower])
x1 = Dense(128, activation='relu')(x1)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(num_classes, activation='softmax', name='x2')(x1)

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]

d1 = Model(inputs=inpu, outputs=x1)

#display(SVG(model_to_dot(model).create(prog='dot', format='svg')))
# load weights
# model.load_weights("weights.forksvhmbest.hdf5")

d1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'],
              callbacks=callbacks)


# In[ ]:

d1.fit(x_train,y1,
        batch_size=batch_size,
        epochs=epochs,
        #validation_split=0.2,
        validation_data=(x_test,y1_t),
        shuffle=True,
        verbose=2,
        callbacks=callbacks)

score = d1.evaluate(x_test, y1_t, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

d1.save('ioncanond1.h5')

print()
print("Digit1 tower complete")
print()

# In[ ]:

numtowerfre = load_model('ioncanonnum.h5')
numtowerfre.name ="Numbertower"
print("starting Digit2")

inpu = Input(shape=x_train.shape[1:])

# Number tower
numtower = numtowerfre(inpu)
numtower.trainable=False

x = Conv2D(32, (2, 2), padding='same')(inpu)
x = Activation('relu')(x)
x = Conv2D(32, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (4, 4), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (4, 4))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

conv_out = Flatten()(x)

x1 = keras.layers.concatenate([conv_out, numtower])
x1 = Dense(128, activation='relu')(x1)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(num_classes, activation='softmax', name='x3')(x1)

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]

d2 = Model(inputs=inpu, outputs=x1)

#display(SVG(model_to_dot(model).create(prog='dot', format='svg')))
# load weights
# model.load_weights("weights.forksvhmbest.hdf5")

d2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'],
              callbacks=callbacks)


# In[ ]:

d2.fit(x_train,y2,
        batch_size=batch_size,
        epochs=epochs,
        #validation_split=0.2,
        validation_data=(x_test,y2_t),
        shuffle=True,
        verbose=2,
        callbacks=callbacks)

score = d2.evaluate(x_test, y2_t, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

d2.save('ioncanond2.h5')

print()
print("Digit2 tower complete")
print()

# In[ ]:

numtowerfre = load_model('ioncanonnum.h5')
numtowerfre.name ="Numbertower"
print("starting Digit3")

inpu = Input(shape=x_train.shape[1:])

# Number tower
numtower = numtowerfre(inpu)
numtower.trainable=False

x = Conv2D(32, (2, 2), padding='same')(inpu)
x = Activation('relu')(x)
x = Conv2D(32, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (4, 4), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (4, 4))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

conv_out = Flatten()(x)


x1 = keras.layers.concatenate([conv_out, numtower])
x1 = Dense(128, activation='relu')(x1)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.5)(x1)


x1 = Dense(num_classes, activation='softmax', name='x1')(x1)

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]

d3 = Model(inputs=inpu, outputs=x1)

#display(SVG(model_to_dot(model).create(prog='dot', format='svg')))
# load weights
# model.load_weights("weights.forksvhmbest.hdf5")

d3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'],
              callbacks=callbacks)


# In[ ]:

d3.fit(x_train,y3,
        batch_size=batch_size,
        epochs=epochs,
        #validation_split=0.2,
        validation_data=(x_test,y3_t),
        shuffle=True,
        verbose=2,
        callbacks=callbacks)

score = d3.evaluate(x_test, y3_t, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

d3.save('ioncanond3.h5')

print()
print("Digit3 tower complete")
print()

# In[13]:

numtowerfre = load_model('ioncanonnum.h5')
numtowerfre.name ="Numbertower"
print("starting Digit4")

inpu = Input(shape=x_train.shape[1:])

# Number tower
numtower = numtowerfre(inpu)
numtower.trainable=False

x = Conv2D(32, (2, 2), padding='same')(inpu)
x = Activation('relu')(x)
x = Conv2D(32, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (4, 4), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (4, 4))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

conv_out = Flatten()(x)

x1 = keras.layers.concatenate([conv_out, numtower])
x1 = Dense(128, activation='relu')(x1)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(num_classes, activation='softmax', name='x4')(x1)

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]

d4 = Model(inputs=inpu, outputs=x1)

# display(SVG(model_to_dot(model).create(prog='dot', format='svg')))
# load weights
# model.load_weights("weights.forksvhmbest.hdf5")

d4.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'],
              callbacks=callbacks)


# In[ ]:

d4.fit(x_train,y4,
        batch_size=batch_size,
        epochs=epochs,
        #validation_split=0.2,
        validation_data=(x_test,y4_t),
        shuffle=True,
        verbose=2,
        callbacks=callbacks)

score = d4.evaluate(x_test, y4_t, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[3]:

print("part2")

X444 = np.load("Xcropresize4.npy")
y444 = np.load("Ycropresize4.npy").astype(int)

y444 = hot(y444)
x_train444, x_test444, y_train444, y_test444 = train_test_split(
    X444, y444, test_size=0.2)
y444 = y_train444[:,33:44]
y444_t= y_test444[:,33:44]

callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0)]

d4.fit(x_train444,y444,
        batch_size=batch_size,
        epochs=50,
        #validation_split=0.2,
        validation_data=(x_test444,y444_t),
        shuffle=True,
        verbose=2,
        callbacks=callbacks)

score = d4.evaluate(x_test444, y444_t, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

d4.save('ioncanond4.h5')

print()
print("Digit4 tower complete")
print()

# In[ ]:

numtowerfre = load_model('ioncanonnum.h5')
numtowerfre.name ="Numbertower"
print("starting Digit5")

inpu = Input(shape=x_train.shape[1:])

# Number tower
numtower = numtowerfre(inpu)
numtower.trainable=False

x = Conv2D(32, (2, 2), padding='same')(inpu)
x = Activation('relu')(x)
x = Conv2D(32, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (2, 2))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (4, 4), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (4, 4))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

conv_out = Flatten()(x)

x1 = keras.layers.concatenate([conv_out, numtower])
x1 = Dense(128, activation='relu')(x1)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(num_classes, activation='softmax', name='x5')(x1)

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]

d5 = Model(inputs=inpu, outputs=x1)

# display(SVG(model_to_dot(model).create(prog='dot', format='svg')))
# load weights
# model.load_weights("weights.forksvhmbest.hdf5")

d5.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'],
              callbacks=callbacks)


# In[ ]:

d5.fit(x_train,y5,
        batch_size=batch_size,
        epochs=epochs,
        #validation_split=0.2,
        validation_data=(x_test,y5_t),
        shuffle=True,
        verbose=2,
        callbacks=callbacks)

score = d5.evaluate(x_test, y5_t, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:
print()
print("part2")
callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0)]
X555 = np.load("Xcropresize5.npy")
y555 = np.load("Ycropresize5.npy").astype(int)

y555 = hot(y555)
x_train555, x_test555, y_train555, y_test555 = train_test_split(
    X555, y555, test_size=0.2)
y555 = y_train555[:,44:55]
y555_t= y_test555[:,44:55]

d5.fit(x_train555,y555,
        batch_size=batch_size,
        epochs=50,
        #validation_split=0.2,
        validation_data=(x_test555,y555_t),
        shuffle=True,
        verbose=2,
        callbacks=callbacks)

score = d5.evaluate(x_test555, y555_t, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

d5.save('ioncanond5.h5')


# In[ ]:

print("MERGING!!")
print()
      
digit1 = load_model('ioncanond1.h5')
digit1.name ="digit1"

digit2 = load_model('ioncanond2.h5')
digit2.name ="digit2"

digit3 = load_model('ioncanond3.h5')
digit3.name ="digit3"

digit4 = load_model('ioncanond4.h5')
digit4.name ="digit4"

digit5 = load_model('ioncanond5.h5')
digit5.name ="digit5"


# In[ ]:

epochs = 3

a = Input(shape=x_train.shape[1:])

# Merging
d1 = digit1(a)

d2 = digit2(a)

d3 = digit3(a)

d4 = digit4(a)

d5 = digit5(a)

model = Model(inputs=a, outputs=[d1, d2, d3, d4, d5])

#display(SVG(model_to_dot(model).create(prog='dot', format='svg')))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[ ]:


A = np.array(model.predict(x_test))
c=0
for i in range(A.shape[1]):
    c+=np.array_equal(np.argmax(np.array(A[:,i,:]).reshape((5,11)),axis=1),Y[i,0:5])
print("Accuracy:",100*c/A.shape[1])


model.fit(x_train,[y1,y2,y3,y4,y5],
        batch_size=batch_size,
        epochs=epochs,
        #validation_split=0.2,
        validation_data=(x_test,[y1_t,y2_t,y3_t,y4_t,y5_t]),
        shuffle=True,
        verbose=2)

A = np.array(model.predict(x_test))
c=0
for i in range(A.shape[1]):
    c+=np.array_equal(np.argmax(np.array(A[:,i,:]).reshape((5,11)),axis=1),Y[i,0:5])
print("Accuracy:",100*c/A.shape[1])

model.save('ioncanondprime.h5')

