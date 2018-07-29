import csv
import io
import os
from typing import Dict, List

import h5py
import numpy as np
import requests
from keras.preprocessing.image import load_img, img_to_array

# This script generate h5 files with labels and images as numpy.array.
#
# Images take from /renders/*/*.jpg, and tags from /table.csv
# Because we have images too lot (currently 54 000 files)
# the h5 file are split by DATA_SLICE constant


CURRENT_DIR: str = os.getcwd()
DATA_SLICE: int = 2
CSV_TABLE = 'https://docs.google.com/spreadsheet/ccc?key=1OCbBLuG_de_Dpb8fxeu0Jcn_ZHWkzrrL7GM4Mfcvgnk&output=csv'
CSV_VALIDATION_TABLE = 'https://docs.google.com/spreadsheet/ccc?key=1DL5ggzlILEQQSkmVXqX1HWMwg0jEPVo8jPIh2Lr7jaA&output=csv'


# Function are collecting tags from CSV_TABLE with test data to Dict
def get_folder_classes():
    tags_data = {}

    response = requests.get(CSV_TABLE)
    assert response.status_code == 200, 'Wrong status code'

    csv_file = io.StringIO(response.text, newline='\r\n')
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)

    for row in reader:
        tags_data[row[0]]: Dict[int] = list(map(int, row[1:-1]))

    return tags_data


# Function are collecting tags from CSV_VALIDATION_TABLE with validation data to Dict
def get_validation_folder_classes():
    tags_data = {}

    response = requests.get(CSV_VALIDATION_TABLE)
    assert response.status_code == 200, 'Wrong status code'

    csv_file = io.StringIO(response.text, newline='\r\n')
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)

    for row in reader:
        tags_data[row[0]]: Dict[int] = list(map(int, row[1:]))

    return tags_data


# Function are collecting information about rendered images to List
# where each item have tags and absolute path to image like this
# [ [[0,1,0,0,1], 'path\to\image.jpg'], [], ... ]
def get_renders_data():
    result = []

    # open csv file with tags and save in dict
    tags_data = get_folder_classes()

    renders_dir = os.path.join(CURRENT_DIR, '..', 'renders')
    dir_list: List[str] = os.listdir(renders_dir)

    for directory_name in dir_list:
        file_list = os.listdir(os.path.join(renders_dir, directory_name))

        for file_name in file_list:
            result.append([
                tags_data[directory_name],
                os.path.join(renders_dir, directory_name, file_name),
            ])

    return result


# Function are collecting information about validation images to List
# where each item have tags and absolute path to image like this
# [ [[0,1,0,0,1], 'path\to\image.jpg'], [], ... ]
def get_validation_data():
    result = []

    # open csv file with tags and save in dict
    tags_data = get_validation_folder_classes()
    images_dir = os.path.join(CURRENT_DIR, '..', 'validation')

    for tag in tags_data:
        result.append([
            tags_data[tag],
            os.path.join(images_dir, tag + '.jpg'),
        ])

    return result


def set_trained_data():
    data = get_renders_data()

    for slice_index in range(DATA_SLICE):
        print('start work slice with', slice_index, 'of', DATA_SLICE)

        test_data_list = []
        test_label_list = []

        train_data_list = []
        train_label_list = []

        index = 0
        for tags, image_name in data[slice_index::DATA_SLICE]:
            image = img_to_array(load_img(image_name, grayscale=True))

            if index % 4 == 0:
                test_data_list.append(image)
                test_label_list.append(tags)

            else:
                train_data_list.append(image)
                train_label_list.append(tags)

            index += 1

        train_data = np.array(train_data_list).astype('uint8')
        train_label = np.array(train_label_list).astype('uint8')
        test_data = np.array(test_data_list).astype('uint8')
        test_label = np.array(test_label_list).astype('uint8')

        slice_str = str(slice_index).zfill(3)
        train_file = os.path.join(CURRENT_DIR, '..', 'data', 'train' + slice_str + '.h5')
        test_file = os.path.join(CURRENT_DIR, '..', 'data', 'test' + slice_str + '.h5')

        with h5py.File(train_file, 'w') as hf:
            hf.create_dataset('train_data', data=train_data)
            hf.create_dataset('train_label', data=train_label)

        with h5py.File(test_file, 'w') as hf:
            hf.create_dataset('test_data', data=test_data)
            hf.create_dataset('test_label', data=test_label)


def set_validation_data():
    validation_data = get_validation_data()
    validation_data_list = []
    validation_label_list = []

    for validation in validation_data:
        image = img_to_array(load_img(validation[1], grayscale=True))

        validation_data_list.append(image)
        validation_label_list.append(validation[0])

    validation_data = np.array(validation_data_list).astype('uint8')
    validation_label = np.array(validation_label_list).astype('uint8')

    validation_file = os.path.join(CURRENT_DIR, '..', 'data', 'validation.h5')
    with h5py.File(validation_file, 'w') as hf:
        hf.create_dataset('data', data=validation_data)
        hf.create_dataset('label', data=validation_label)


set_trained_data()
set_validation_data()
