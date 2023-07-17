# Neural networks for image recognition

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)

## General Information
The work presents the process of image identification and classification into several categories. For the purpose of conducting the study, a Python project was prepared, which properly formats the dataset, trains the neural network, and simultaneously creates its model. By classifying images into several predefined categories, the program is able to determine what is depicted in a given image. The dataset chosen for this project consists of images depicting popular species of characteristic flowers, available at: [Flowers Recognition | Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition). The neural network model has been tailored to this dataset, and the network analysis was conducted based on this dataset. Additionally, considering that both shape and color are important in flower classification, the neural network model utilizes color images.

In the project, it is easy to change parameters such as the number of epochs, activation function, or optimizer to see in which case the model performs best. We can observe the effects on the plots.

## Technologies Used

1. **NumPy**: A library for numerical computing in Python, used for mathematical operations and array manipulation.

2. **OpenCV (cv2)**: A computer vision library that provides tools for image and video processing, used here for loading and manipulating images.

3. **Shutil**: A module in Python used for high-level file operations, such as copying files and directories.

4. **Random (rn)**: A module in Python used for generating random numbers and performing random operations, used here for shuffling the data.

5. **Tqdm**: A library for creating progress bars in Python, used here for visualizing the progress of loops.

6. **Matplotlib**: A plotting library in Python, used here for displaying images and creating plots to visualize the training and validation results.

7. **TensorFlow**: An open-source machine learning framework, used here for building, training, and evaluating neural network models.

8. **Keras**: A high-level API built on top of TensorFlow, used here for simplifying the process of defining and training neural networks.

## Features

List the ready features here:

1. Loading training data and validation data using `tf.keras.preprocessing.image_dataset_from_directory` function.
2. Displaying sample images from the training data using `matplotlib.pyplot.imshow`.
3. Data normalization using `layers.experimental.preprocessing.Rescaling` to scale pixel values in the range [0, 1].
4. Model definition using `Sequential` model from TensorFlow.
5. Adding layers to the model, including convolutional layers (`Conv2D`), pooling layers (`MaxPooling2D`), and fully connected layers (`Dense`).
6. Compiling the model with optimizer, loss function, and metrics using `model.compile`.
7. Training the model using the `fit` function, providing training data, validation data, and number of epochs.
8. Displaying training and validation accuracy and loss using `matplotlib.pyplot.plot` and `matplotlib.pyplot.title`.

Additionally, the code includes data augmentation techniques for image augmentation:

1. Using `layers.experimental.preprocessing.RandomFlip` for horizontal flipping of images.
2. Using `layers.experimental.preprocessing.RandomRotation` for random rotation of images.
3. Using `layers.experimental.preprocessing.RandomZoom` for random zooming of images.

The augmented images are then used to train the model with data augmentation, and the training and validation accuracy and loss are displayed for the augmented model.

## Screenshots

In the code, sample images from the training dataset were displayed along with their labels.

![Example screenshot](./screens/flowers.png)

![Example screenshot](./screens/chart1.png)

Plots for the number of epochs equal to 20.

![Example screenshot](./screens/plots2.png)

Plots after changing the activation function to tanh.

![Example screenshot](./screens/plots3.png)

## Setup

The project requirements/dependencies are as follows:

1. numpy
2. opencv-python
3. tqdm
4. matplotlib
5. tensorflow

To set up your local environment and get started with the project, follow these steps:

1. Install Python: Make sure you have Python installed on your computer. 

2. Install the required dependencies: Open a command prompt or terminal and run the following commands to install the necessary dependencies:

   ```
   pip install numpy opencv-python tqdm matplotlib tensorflow
   ```

   This will install the required libraries and frameworks for the project.

3. Set up the data directory: Update the `data_dir` variable in the code to the path where your flower dataset is located. Make sure the dataset is organized in a directory structure where each class of flowers is in a separate subdirectory.

4. Run the code: Save the code in a Python file with a `.py` extension (e.g., `flower_recognition.py`). Open a command prompt or terminal, navigate to the directory where the file is saved, and run the following command:

   ```
   python flower_recognition.py
   ```

   This will execute the code and train the neural network model for flower recognition. The training progress and results will be displayed in the console and as plots using matplotlib.

## Usage

Loading training and validation data from the directory:

 ```
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
```

Normalizing the data and setting up data augmentation:

```
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

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
```

Defining the model architecture:

```
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
```
Compiling and training the model:

```
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

epochs = 10

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

