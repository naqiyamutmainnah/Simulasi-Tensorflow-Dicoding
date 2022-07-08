# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil

def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    TRAINING_DIR = "data/rps/"
    train_dir = os.path.join(TRAINING_DIR, 'train')
    validation_dir = os.path.join(TRAINING_DIR, 'val')
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    rock_dir = os.path.join(TRAINING_DIR, 'rock')
    paper_dir = os.path.join(TRAINING_DIR, 'paper')
    scissors_dir = os.path.join(TRAINING_DIR, 'scissors')

    train_rock_dir, val_rock_dir = train_test_split(os.listdir(rock_dir), test_size=0.2)
    train_paper_dir, val_paper_dir = train_test_split(os.listdir(paper_dir), test_size=0.2)
    train_scissors_dir, val_scissors_dir = train_test_split(os.listdir(scissors_dir), test_size=0.2)

    train_rock = os.path.join(train_dir, 'rock')
    train_paper = os.path.join(train_dir, 'paper')
    train_scissors = os.path.join(train_dir, 'scissors')
    val_rock = os.path.join(validation_dir, 'rock')
    val_paper = os.path.join(validation_dir, 'paper')
    val_scissors = os.path.join(validation_dir, 'scissors')

    os.mkdir(train_rock)
    os.mkdir(train_paper)
    os.mkdir(train_scissors)
    os.mkdir(val_rock)
    os.mkdir(val_paper)
    os.mkdir(val_scissors)

    for data in train_rock_dir:
        shutil.copy(os.path.join(rock_dir, data), os.path.join(train_rock, data))
    for data in train_paper_dir:
        shutil.copy(os.path.join(paper_dir, data), os.path.join(train_paper, data))
    for data in train_scissors_dir:
        shutil.copy(os.path.join(scissors_dir, data), os.path.join(train_scissors, data))

    for data in val_rock_dir:
        shutil.copy(os.path.join(rock_dir, data), os.path.join(val_rock, data))
    for data in val_paper_dir:
        shutil.copy(os.path.join(paper_dir, data), os.path.join(val_paper, data))
    for data in val_scissors_dir:
        shutil.copy(os.path.join(scissors_dir, data), os.path.join(val_scissors, data))

    training_datagen = ImageDataGenerator(
        # YOUR CODE HERE
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255)

    train_generator = training_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )  # YOUR CODE HERE

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )


    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy']
                  )
    model.fit(train_generator,
              epochs=20,
              verbose=2,
              validation_data=validation_generator,
              )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
