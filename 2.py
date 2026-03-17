# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
print("Training data:")
print(X.shape)
print(y.shape)
print("Classes:")
print(np.unique(y))
print("Number of unique words:")
print(len(np.unique(np.hstack(X))))
print("Review length:")
result = [len(x) for x in X]
print("Mean %.2f words (Std: %.2f)" % (np.mean(result), np.std(result)))
plt.boxplot(result)
plt.title("Review Length Distribution")
plt.ylabel("Number of Words")
plt.show()