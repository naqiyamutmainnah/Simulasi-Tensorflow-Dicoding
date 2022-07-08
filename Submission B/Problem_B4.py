# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    x = bbc.text
    y = bbc.category

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        train_size=training_portion,
                                                        shuffle=False
                                                        )

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words = vocab_size,
                           oov_token = oov_tok
                          )

    tokenizer.fit_on_texts(x_train)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(train_sequences,
                            maxlen=max_length,
                            truncating=trunc_type,
                            padding=padding_type
                            )

    test_sequences = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(test_sequences,
                           maxlen=max_length,
                           truncating=trunc_type,
                           padding=padding_type
                           )

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(y)

    y_train = np.array(label_tokenizer.texts_to_sequences(y_train))
    y_test = np.array(label_tokenizer.texts_to_sequences(y_test))

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['acc']
                 )

    model.fit(x_train,
              y_train,
              epochs=50,
              validation_data=(x_test, y_test)
             )

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")