# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 20:07:25 2021

@author: sushanthsgradlaptop2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
mnist=tf.keras.datasets.mnist
(train_data,train_labels),(test_data,test_labels)=mnist.load_data()
train_data=train_data/255.0
test_data=test_data/255.0
train_data_m=np.mean(train_data,axis=0)
train_data_max=np.max(train_data,axis=0)
train_data_min=np.min(train_data,axis=0)
train_data_n=(train_data-train_data_m)/(1e-4+train_data_max+train_data_min)
test_data_n=(test_data-train_data_m)/(1e-4+train_data_max+train_data_min)
def res_blk(inputs,num_fils,kern_size,strides):
	x=tf.keras.layers.BatchNormalization()(inputs)
	x=tf.keras.layers.Conv2D(filters=num_fils,kernel_size=kern_size,strides=strides,
						  activation='linear',padding='same')(x)
	x=tf.keras.layers.ReLU()(x)
	x=tf.keras.layers.BatchNormalization()(x)
	x=tf.keras.layers.Conv2D(filters=num_fils,kernel_size=kern_size,strides=strides,
						  activation='linear',padding='same')(x)
	x=tf.keras.layers.Add()([x,inputs])
	x=tf.keras.layers.ReLU()(x)
	return(x)

Ip=tf.keras.layers.Input(shape=(28,28,1,))
xm=res_blk(Ip,48,(5,5),(1,1))
gap=tf.keras.layers.GlobalAveragePooling2D()(xm)
op_layer=tf.keras.layers.Dense(units=10,activation='softmax')(gap)
mod_1=tf.keras.Model(inputs=Ip,outputs=op_layer)
mod_1.summary()
#%%
#ot_model(mod_1,show_dtype=True,show_shapes=True)

mod_1.compile(loss='sparse_categorical_crossentropy',
			  optimizer=tf.keras.optimizers.Adam(),
			  metrics=['sparse_categorical_accuracy'])
his=mod_1.fit(train_data_n,
			  train_labels,validation_data=(test_data_n,
														 test_labels),epochs=10,batch_size=64)
#%%
epochs=10
plt.plot(range(epochs),his.history['loss'],range(epochs),his.history['val_loss'])
plt.figure()
plt.plot(range(epochs),his.history['sparse_categorical_accuracy'],
		 range(epochs),his.history['val_sparse_categorical_accuracy'])
#%%
wm=mod_1.layers
wm_c=wm[2].weights[0].numpy().squeeze()
cm=np.concatenate([wm_c[:,:,j].reshape(5,5) for j in range(4)],axis=1)
plt.imshow(cm,'gray')
		
