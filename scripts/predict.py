import os

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input

SIZE: int = 150

MODEL = os.path.join(os.getcwd(), '..', 'models', 'main.h5')
VALID_EXT_IMAGES = [
    os.path.join(os.getcwd(), '..', 'validation', '001.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '002.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '003.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '004.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '005.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '006.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '007.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '008.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '009.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '010.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '011.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '012.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '013.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '014.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '015.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '016.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '017.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '018.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '019.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '020.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '021.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '022.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '023.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '024.jpg'),
    os.path.join(os.getcwd(), '..', 'validation', '025.jpg'),
    ]

# armrests	wheels	rotating	back	footrest	height adjustment
LABEL_DICT = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
]


def decode_predictions(preds, top=5):
    results = []

    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]

        result = [tuple([i, bin(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)

        results.append(result)

    return results


model = load_model(MODEL)

# Score trained model.
for img in VALID_EXT_IMAGES:
    x = img_to_array(load_img(img, grayscale=False, target_size=(SIZE, SIZE)))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Prediction        ', img)
    # print(decode_predictions(preds)[0][0])
    print('подлокотники      ', round(preds[0][0], 2))
    print('колесики          ', round(preds[0][1], 2))
    print('вращается         ', round(preds[0][2], 2))
    print('есть спинка       ', round(preds[0][3], 2))
    print('подставка для ног ', round(preds[0][4], 2))
    print('регулировка высоты', round(preds[0][5], 2))
    print('')
