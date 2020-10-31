import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

train_img = train_img.astype('float32')
test_img = test_img.astype('float32')

train_img = train_img / 255.0
test_img = test_img / 255.0

train_x = train_img.tolist()
train_y = train_labels.tolist()

# shift left, right, up and down by one pixel

def shift_vertical(OFFSET):
  for i in range(train_img.shape[0]):
    timg = train_img[i]
    img = np.zeros([28, 28])
    for x in range(28):
      img[x] = timg[(x + OFFSET + 28) % 28]
    train_x.append(img)
    train_y.append(train_labels[i])

def shift_horizontal(OFFSET):
  for i in range(train_img.shape[0]):
    timg = train_img[i]
    img = np.zeros([28, 28])
    for y in range(28):
      for x in range(28):
        img[x][y] = timg[x][(y + OFFSET + 28) % 28]
    train_x.append(img)
    train_y.append(train_labels[i])

# right
shift_horizontal(-2)
# left
shift_horizontal(2)
# up
shift_vertical(2)
# down
shift_vertical(-2)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_img = test_img.reshape(test_img.shape[0], 28, 28, 1)

print("Dataset expansion done..!")
print(train_x.shape, train_y.shape)

# network

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, kernel_size=(2, 2), input_shape=(13, 13, 32)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(2304, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


model.compile(
  optimizer='SGD',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

# train
model.fit(train_x, train_y, epochs=30, batch_size=10)

# evaluate
test_loss, test_acc = model.evaluate(test_img, test_labels, verbose=2)
print("Accuracy: {0:.2%}".format(test_acc))
