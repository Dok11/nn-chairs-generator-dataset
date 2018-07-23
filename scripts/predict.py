import os

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img


MODEL = os.path.join(os.getcwd(), '..', 'models', 'main.h5')
VALID_EXT_IMAGES = [
    os.path.join(os.getcwd(), '..', 'validation', 'ext.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', 'ext2.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', 'ext3.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', 'ext4.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '020.jpg'),
    ]

model = load_model(MODEL)

# Score trained model.
for img in VALID_EXT_IMAGES:
    x = img_to_array(load_img(img, grayscale=True))
    x = np.expand_dims(x, axis=0)
    scores = model.predict(x)
    print('Prediction        ', img)
    print('подлокотники      ', round(scores[0][0], 2))
    print('колесики          ', round(scores[0][1], 2))
    print('вращается         ', round(scores[0][2], 2))
    print('есть спинка       ', round(scores[0][3], 2))
    print('подставка для ног ', round(scores[0][4], 2))
    print('регулировка высоты', round(scores[0][5], 2))
    print('')
