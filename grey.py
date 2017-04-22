
# coding: utf-8

# In[8]:

import keras
import numpy as np
from scipy.misc import imresize
from matplotlib import pyplot as plt


# In[23]:

class SVHN():

    path = ""

    def __init__(self,model):
        self.model=model

    def get_sequence(self, image):
        A = list(self.model.predict(imresize(image,(50,50),interp='cubic').reshape((1,50,50,3))))
        return([x%10 for x in np.argmax(np.array(A).reshape((6,11)),axis=1) if x != 0])
    
    @staticmethod
    def load_model():
        loadModel=keras.models.load_model("forksvhmaug.h5")
        svhnObj=SVHN(loadModel)
        return svhnObj
        

if __name__ == "__main__":
        obj=SVHN.load_model()
        X = np.load("resize_cropImage.npy")[0:13000,:]
        num = np.random.randint(len(X))
        plt.imshow(X[num])
        plt.show()
        print(obj.get_sequence(X[num]))
        print(nummertrain[num])


# In[19]:

nummertrain=np.load("nummertrain.npy") 


# In[ ]:

nummertrain

