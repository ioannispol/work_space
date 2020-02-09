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

# Evaluate accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


# TODO Overfitting, what is it and how can be prevented

# Make predictions

predictions = model.predict(test_images)
predictions[0]
print(predictions)

np.argmax(predictions[0])
test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_cat[predicted_label],
                                100*np.max(predictions_array),
                                class_cat[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Verify predictions

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and their true labels.
# color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()
