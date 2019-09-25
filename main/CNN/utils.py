import os
import json
import numpy as np
from PIL import Image


def make_teacher_data(base_path, teacher_directory_list, split_data=0.1):
    train_data = []
    val_data = []
    for teacher_directory, counter in zip(teacher_directory_list, range(len(teacher_directory_list))):
        teacher_directory_path = base_path + teacher_directory

        directory_contain_list = os.listdir(teacher_directory_path)

        teacher_data = []
        for directory_contain in directory_contain_list:
            image_path = teacher_directory_path + '\\' + directory_contain
            label = [1 if i == counter else 0 for i in range(len(teacher_directory_list))]
            # laber_str = ','.join(label)
            teacher_data.append([image_path, label])

        num_val, num_train = split_train_data(teacher_data, split_data)
        train_data.extend(teacher_data[:num_train])
        val_data.extend(teacher_data[num_train:])
    return train_data, val_data


def scale_change(image, aspect_width, aspect_height):
    """resize image with unchanged aspect ratio using padding"""
    img_width = image.size[0]
    img_height = image.size[1]

    if img_width > img_height:
        rate = aspect_width / img_width
    else:
        rate = aspect_height / img_height
    image = image.resize((int(img_width * rate), int(img_height * rate)))
    iw, ih = image.size
    new_image = Image.new('RGB', (aspect_width, aspect_height), (128, 128, 128))
    new_image.paste(image, ((aspect_width - iw) // 2, (aspect_height - ih) // 2))
    return new_image


def data_generator(data_list, batch_size, image_shape=(224, 224)):
    """data generator for fit_generator"""
    n = len(data_list)
    i = 0
    width = image_shape[0]
    height = image_shape[1]
    while True:
        image_data = []
        label = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(data_list)
            try:
                image_array = image_open(data_list[i][0], width, height)
            except:
                print('image not opne')
                continue
            if image_array.ndim != 3:
                continue
            image_data.append(image_array)
            label.append(data_list[i][1])
            i = (i + 1) % n
        image_data = np.array(image_data)
        label = np.array(label)
        yield image_data, label


def image_open(image_path, width, height):
    image = Image.open(image_path)
    image = scale_change(image, width, height)
    image_array = np.asarray(image)
    return image_array


def split_train_data(data_list, val_split):
    num_val = int(len(data_list) * val_split)
    num_train = len(data_list) - num_val
    return num_val, num_train
