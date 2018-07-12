import os

import h5py
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD


BATCH_SIZE = 32
NUM_CLASSES = 9
EPOCHS = 20
SAVE_DIR = os.path.join(os.getcwd(), '..', 'models', 'main.h5')


def get_dataset():
    test_file_path = os.path.join(os.getcwd(), '..', 'data', 'test000.h5')
    train_file_path = os.path.join(os.getcwd(), '..', 'data', 'train000.h5')

    with h5py.File(test_file_path, 'r') as hf:
        test_data = hf['test_data'][:]
        test_label = hf['test_label'][:]

    with h5py.File(train_file_path, 'r') as hf:
        train_data = hf['train_data'][:]
        train_label = hf['train_label'][:]

    return (train_data, train_label), (test_data, test_label)


(x_train, y_train), (x_test, y_test) = get_dataset()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 256 -> 128
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 128 -> 64
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 64 -> 32
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(NUM_CLASSES * 64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))

sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
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

# Save model and weights
if scores[1] > 0.9:
    model.save(SAVE_DIR)
    print('Saved trained model at %s ' % SAVE_DIR)
