import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
vocab_size = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
print(x_train[0]) 

word_idx = imdb.get_word_index()

word_idx = {i: word for word, i in word_idx.items()}

print([word_idx.get(i, '?') for i in x_train[0]])

print("Max length of a review:: ",
 len(max((x_train.tolist() + x_test.tolist()), key=len)))
print("Min length of a review:: ",
 len(min((x_train.tolist() + x_test.tolist()), key=len)))

max_words = 400
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)

x_valid, y_valid = x_train[:64], y_train[:64]
x_train_, y_train_ = x_train[64:], y_train[64:]


embd_len = 32

RNN_model = Sequential(name="Simple_RNN")

RNN_model.add(Embedding(vocab_size, embd_len, input_length=max_words))

RNN_model.add(SimpleRNN(128, activation='tanh'))

RNN_model.add(Dense(1, activation='sigmoid'))

print(RNN_model.summary())

RNN_model.compile(
 loss="binary_crossentropy",
 optimizer='adam',
 metrics=['accuracy']
) 
history = RNN_model.fit(
 x_train_,
 y_train_,
 batch_size=64,

 epochs=5,
 verbose=1,
 validation_data=(x_valid, y_valid)
)

print()
print("Simple_RNN Score---> ",
 RNN_model.evaluate(x_test, y_test, verbose=0))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show() 
