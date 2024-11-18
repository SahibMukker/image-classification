import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow
from tensorflow import keras
from keras import datasets, layers, models

'''
Getting and preparing data
'''
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Ccaling down data so all values are between 0 and 1
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

'''
Visualizing images, moving this to comment code for now but keeping it instead of deleting for future reference if needed

    
# 4x4 grid, each iteration we are choosing a place in the grid to put next point
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    
    # showing first 16 images with a binary colour map
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
    
plt.show()
'''

'''
# Reducing amount of images feeding into the neural network if needed
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_images = testing_images[:4000]
'''


'''
Building the neural network

Convolutional layers will determine features (ex. horse has long legs, truck is bigger than car, etc.)
'''

'''
Now that model has been trained, going to put all this code into 
comment code since it is not needed (can remove it but i will keep it for future reference if needed)

model = models.Sequential()

# Input layer, 32 neurons in this layer, the convultion matrix(filter) is 3x3, activation is relu, input shape is 32x32x3 (size of image 32x32 and 3 colour ways (RGB))
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
# Simplifies result to essential information into 2x2 mutually exclusive blocks and outputs the max value from each block
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs = 10, validation_data = (testing_images, testing_labels))

# Saving model
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'Loss: {loss}\nAccuracy: {accuracy}')

model.save('image_classifier.keras')
'''
model = models.load_model('image_classifier.keras')

# Will use images from online to get random images to test our model from 'pixabay', a license free image website
img = cv.imread('plane.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

# This has 10 activtions of the 10 softmax neurons, want to have maximum value and index of that prediction
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')

# To see what the actual image is
plt.show()