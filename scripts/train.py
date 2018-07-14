import os

import h5py
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model
from keras.optimizers import SGD


BATCH_SIZE = 196
NUM_CLASSES = 6
EPOCHS = 50
SAVE_DIR = os.path.join(os.getcwd(), '..', 'models', 'main.h5')
USE_SAVED_MODEL: bool = False
SLICE_COUNT: int = 15
SLICE_PART: str = '000'


def get_dataset(slice_part: str):
    test_file_path = os.path.join(os.getcwd(), '..', 'data', 'test' + slice_part + '.h5')
    train_file_path = os.path.join(os.getcwd(), '..', 'data', 'train' + slice_part + '.h5')

    with h5py.File(test_file_path, 'r') as hf:
        test_data = hf['test_data'][:]
        test_label = hf['test_label'][:]

    with h5py.File(train_file_path, 'r') as hf:
        train_data = hf['train_data'][:]
        train_label = hf['train_label'][:]

    return (train_data, train_label), (test_data, test_label)


if USE_SAVED_MODEL:
    model = load_model(SAVE_DIR)

    for x in range(SLICE_COUNT):
        print('Start train', x + 1, 'from', SLICE_COUNT)
        (x_train, y_train), (x_test, y_test) = get_dataset(str(x).zfill(3))

        model.fit(x_train, y_train,
                  verbose=2,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(x_test, y_test),
                  shuffle=True)

else:
    (x_train, y_train), (x_test, y_test) = get_dataset(SLICE_PART)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = Sequential()

    model.add(Conv2D(4, (5, 5), input_shape=x_train.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 254 -> 128

    model.add(Conv2D(16, (4, 4), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 127 -> 64
    model.add(Dropout(0.25))

    model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 63 -> 32
    model.add(Dropout(0.25))

    model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 31 -> 16
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(NUM_CLASSES * 64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              verbose=2,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              shuffle=True)


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.summary()

# Save model and weights
if scores[1] > 0.7:
    model.save(SAVE_DIR)
    print('Saved trained model at %s ' % SAVE_DIR)
