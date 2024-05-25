
#A Multi Layer perceptron Neural Network



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#importing all libraries and frameworks
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation 


#splitting and loading dataset

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()



#printing feature and target matrix
print('Feature Matrix:',x_train.shape)
print('Target Matrix:',x_test.shape)
print('Feature Matrix:',y_train.shape)
print('Target Matrix:',y_test.shape)


#Data Visualization


fig, ax =plt.subplots(10,10)
k =0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train[k].reshape(28,28),aspect='auto')
        k +=1
plt.show()                        
                 
 
                                 
#Multi layer perceptron  of neural network

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(256,activation='sigmoid'),
    Dense(128,activation='sigmoid'),
    Dense(10,activation='sigmoid'),
])

                                               
                                                                   #compilation of model

model.compile(optimizer='adam',
              loss ='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#Evaluation of the model
                            
model.fit(x_train,y_train,epochs=10,
          batch_size=2000,
          validation_split=0.2)              
          
          
#This model is programmed by Dev Kumar.          
                                                                 
