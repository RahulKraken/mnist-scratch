import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# load dataset
mnist = tf.keras.datasets.mnist
(train_img, train_label), (test_img, test_label) = mnist.load_data()

# scale dataset
train_img = train_img / 255.0
test_img = test_img / 255.0

'''
# build the model
step 1: configure the layers
step 2: compile
'''
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(30, activation='relu'),
  tf.keras.layers.Dense(10)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

# train
model.fit(train_img, train_label, epochs=10)

# evaluate
test_loss, test_acc = model.evaluate(test_img, test_label, verbose=2)
print('\nTest acc:', test_acc)
