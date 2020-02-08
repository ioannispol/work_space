from __future__ import division, absolute_import, print_function, unicode_literals

# Common inputs
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow and keras inputs
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Import the MNIST dataset
mnist_dataset = tf.keras.datasets.fashion_mnist

# Load the traing and test images and labes
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

''' Loading the dataset results in a numpy array where the:
* train_images and train_labels are the traing set
* test_images and test_labels are the test set
the model (the train set) is tested against the test set '''

# The ten classes in the MNIST dataset are

class_cat = [
'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Explore the data
train_images.shape
len(train_labels)
train_labels

test_images.shape
len(test_labels)
test_labels

''' Preprocess the data '''

plt.figure()
plt.imshow(train_images[10])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_cat[train_labels[i]])
plt.show()


''' Build the Model. To build the neural network requires to configuarate the layers
of the model and then to compile it.'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Most of deep learning consists of chaining together simple layers.
# Most layers such as keras.layers.Dense, have parameters that are learnd
# during traing.
''' TODO ---> learn: layers, models, variables '''

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

#To commit
