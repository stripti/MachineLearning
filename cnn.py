# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution(adding a convolution layer to the network)
# Here, we take 32 as nb_filter because we are working on a CPU and not on GPU so it will take
# time to process..and no of rows and col of feature detector as 3 and 3...
# input_shape helps us to change all images to a particular size here we take as 64*64 and 3
#  as there are 3 dimension array as colors are taken into account(RGB)

classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
# we choose mostly 2*2 as the size of pool because we dont wanna loose the information and
# still be precise about the max value in the map that corresponds to the feature matched

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
# we dont need input shape coz here inputs are the pooled feature maps from previous step
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# in this step we take output_dim i.e the no of hidden layers as a choice by experiments as 128 ..
# its a no not too small to make classifier good model not too big to make it highly compute intensive..
# so keep it near 100 but a power of 2
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# target_size is same as the input shape of images and class mode is binary coz we have 2 categories cats and dogs..
training_set = train_datagen.flow_from_directory('dataset for cnn/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset for cnn/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# samples per epoch is no of images in training set and nb_val_samples is no of images in test set
classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         nb_val_samples=2000)
