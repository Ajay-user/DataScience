import tensorflow_datasets as tf_ds
import numpy as np
import logging
import os


# hardcoding data directory
path_to_images = './data/images.npy'
path_to_labels = './data/labels.npy'


def download_data():

    # download the data
    data = tf_ds.load('cifar10_1/v6', split='test')
    logging.info('cifar dataset : cifar10_1/v6 downloading completed')

    images, labels = [], []

    for example in data:
        images.append(example['image'].numpy())
        labels.append(example['label'].numpy())

    images = np.array(images)
    labels = np.array(labels)

    logging.info(
        f'dataset :\n Shape of image, {images.shape}\n Shape of label, {labels.shape}')

    with open(path_to_images, mode='wb') as f:
        np.save(f, images)
    with open(path_to_labels, mode='wb') as f:
        np.save(f, labels)

    logging.info('Writing images and labels to file completed')
    logging.info('Data directory "./data/.."')

    return path_to_images, path_to_labels


def download_util():
    if os.path.exists('./data/images.npy'):
        print('Data exists in directory "./data"')
        return path_to_images, path_to_labels
    else:
        return download_data()


def load_data():
    img_src, label_src = download_util()
    logging.info('Loading data..')
    with open(img_src, 'rb') as img, open(label_src, 'rb') as label:
        images = np.load(img)
        labels = np.load(label)
    return images, labels


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    download_util()
