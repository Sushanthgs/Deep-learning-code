# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:40:35 2021

@author: sushanthsgradlaptop2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
mnist=tf.keras.datasets.mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train/255.0
X_test=X_test/255.0

K.clear_session()
def create_model():
	Ip=tf.keras.layers.Input(shape=(28,28,1,))
	b1=tf.keras.layers.BatchNormalization()(Ip)
	x=tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same')(b1)
	x=tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation=None,padding='same')(x)
	x=tf.keras.layers.Add()([b1,x])
	x=tf.keras.layers.ReLU()(x)
	xp=tf.keras.layers.MaxPooling2D()(x)
	xp_g=tf.keras.layers.GlobalAveragePooling2D()(xp)
	op_d=tf.keras.layers.Dense(units=64,activation='softmax')(xp_g)
	mod=tf.keras.Model(inputs=Ip,outputs=op_d)
	return(mod)

m1=create_model()
plot_model(m1)
#%%
m1.compile(loss='sparse_categorical_crossentropy',
		   optimizer=tf.keras.optimizers.Adam(),
		   metrics='accuracy')
hist=m1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=32)
#%%
epochs=10
plt.plot(range(epochs),hist.history['loss'],range(epochs),hist.history['val_loss'])
plt.figure()
plt.plot(range(epochs),hist.history['accuracy'],range(epochs),hist.history['val_accuracy'])
#%%
plot_model(m1)