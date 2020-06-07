# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
model = tf.keras.Sequential([tf.keras.layers.Dense(input_shape=(1,), units = 1)])
model.compile('adam', 'mse')
x = np.array([1,2,3,4,5])
y = np.array([10, 20, 30, 40, 50])
model.fit(x, y, epochs=5000)
tf.keras.sa