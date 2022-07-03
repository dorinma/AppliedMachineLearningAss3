import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(
    'C:\\Users\\shaha\\PycharmProjects\\Ass3\\mnist_784.csv')  # You need to change #directory accordingly
dataset.head(10)  # Return 10 rows of data

# Neural network
model = Sequential()
model.add(Dense(500, input_dim=28 * 28, activation='sigmoid'))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# dataset import

# dataset = pd.read_csv('dataa/mnist_train.csv')
X = dataset.iloc[:, :28 * 28].values
y = dataset.iloc[:, 28 * 28:28 * 28 + 1].values
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
history = model.fit(X_train, y_train, epochs=50, batch_size=100)

y_pred = model.predict(X_test)
# Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
# Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

a = accuracy_score(pred, test)
b = log_loss(test, y_pred)
print('Accuracy is:', a * 100)
print('loss is:', b)

plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
