import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf

from keras.applications import VGG19
from keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.callbacks import Callback

from keras import Sequential
from keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D


full_dataset, dataset_info = tfds.load('oxford_flowers102', split='train+test+validation', as_supervised=True,
                                       with_info=True)

IMG_SIZE = 244
IMG_NORM = 255.0
BATCH_SIZE = 32
acc_ = []
loss_ = []


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        loss, acc = self.model.evaluate(self.test_data, verbose=0)
        acc_.append(acc)
        loss_.append(loss)


def get_data(dataset, train_size, validation_size, shuffle=False):
    train_size = int(train_size * len(dataset))
    validation_size = int(validation_size * len(dataset))

    if shuffle:
        dataset = dataset.shuffle(len(dataset), seed=shuffle)

    ds_train = dataset.take(train_size)
    ds_valid = dataset.skip(train_size)
    ds_valid = dataset.take(validation_size)
    ds_test = dataset.skip(train_size + validation_size)

    print(f"Total images: {len(dataset)}.")
    print(f"Train: {len(ds_train)}, test: {len(ds_test)}, validation: {len(ds_valid)}.")

    return ds_train, ds_test, ds_valid


def reformat_img(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)) / IMG_NORM
    return img, label


def preprocess(dataset, train_size):
    dataset = dataset.shuffle(train_size).map(reformat_img)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = dataset.prefetch(1)
    return dataset


def create_vgg_model():
    model = VGG19(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    for layer in model.layers[:19]:
        layer.trainable = False
    return model


def create_inception_v3_model():
    model = InceptionV3(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in model.layers[:-1]:
        layer.trainable = False
    return model


def create_model(num_classes, model_not_trained):
    model = Sequential()
    model.add(model_not_trained)
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    ds_train, ds_test, ds_valid = get_data(full_dataset, train_size=0.6, validation_size=0.2)

    train_size = len(ds_train)
    ds_train_processed = preprocess(ds_train, train_size)
    ds_test_processed = preprocess(ds_test, train_size)
    ds_valid_processed = preprocess(ds_valid, train_size)

    num_classes = dataset_info.features['label'].num_classes

    model = create_model(num_classes, create_inception_v3_model())
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1,
                                                                factor=0.6, min_lr=0.000001)
    epochs = 10
    history = model.fit(ds_train_processed, epochs=epochs, validation_data=ds_valid_processed, verbose=1,
                        callbacks=[TestCallback(ds_test_processed), learning_rate_reduction, early_stopping])

    res = model.evaluate(ds_test_processed)
    print("[Loss, Accuracy] = ", res)

    # plot the accuracy graph
    epochs = list(range(1, epochs + 1))
    plt.plot(epochs, history.history['accuracy'], label="Training")
    plt.plot(epochs, history.history['val_accuracy'], label="Validation")
    plt.plot(epochs, acc_, label="Testing")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig("Accuracy.png")

    # plot the loss graph
    plt.plot(epochs, history.history['loss'], label="Training")
    plt.plot(epochs, history.history['val_loss'], label="Validation")
    plt.plot(epochs, loss_, label="Testing")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig("Loss.png")


main()
