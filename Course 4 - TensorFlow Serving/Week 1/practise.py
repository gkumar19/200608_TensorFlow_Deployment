# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:44:09 2020

@author: Gaurav
"""
import tensorflow as tf

#%%
tf.saved_model.save
tf.keras.experimental.saved_model()


#%%
import tensorflow_hub as hub
import tensorflow as tf
embed = hub.load("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1")

#%%

m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", output_shape=[1001])
])
m.build([None, 224, 224, 3])  # Batch input shape.

m.summary()
