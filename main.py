# Importing necessary libraries and frameworks
import numpy as np
import os
import cv2
import shutil
import random as rn
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Description of the overall code
# This code presents an example of training a neural network model for flower recognition.
# It utilizes the TensorFlow framework for building and training the model, and the matplotlib library for displaying the results.

# Path to the data directory
data_dir = "C:\\Users\\Uzytkownik\\Downloads\\archive\\flowers"

# Display the list of files in the data directory
print(os.listdir("C:\\Users\\Uzytkownik\\Downloads\\archive\\flowers"))

# Parameters: batch_size, img_height, and img_width
batch_size = 32
img_height = 180
img_width = 180

# Loading training data from the directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Loading validation data from the directory
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Class names
class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

# Displaying sample images from the training data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

    # Displaying the shape of one batch of training data
    for image_batch, labels_batch in train_ds:
      print(image_batch.shape)
      print(labels_batch.shape)
      break

AUTOTUNE = tf.data.AUTOTUNE

# Optimization and buffering of training and validation data
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data normalization layer
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# Applying normalization to the training data
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Pixel values are now in the range [0,1]
print(np.min(first_image), np.max(first_image))

num_classes = 5

# Model definition
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compiling the model
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

epochs = 10

# Training the model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Displaying training and validation accuracy in each epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Displaying training and validation loss values in each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Training and validation accuracy plots
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Data augmentation
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

# Displaying sample images after applying data augmentation
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

    # Model definition with data augmentation
    model = Sequential([
      data_augmentation,
      layers.experimental.preprocessing.Rescaling(1. / 255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.3),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
    )

    epochs = 10

    # Training the model with data augmentation
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

# Displaying training and validation accuracy with data augmentation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Displaying training and validation loss values with data augmentation
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Training and validation accuracy plots with data augmentation
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
