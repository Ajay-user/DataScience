
import logging
import numpy as np


# util
from utils import score_fn
from loading_data import load_data
from Nearest_Neighbours import NearestNeighbours
from K_Nearest_Neighbours import KNearestNeighbours


logging.basicConfig(level=logging.INFO)


# download the data
images, labels = load_data()

# train-test split
train_images = images[:1500]
test_images = images[1500:]
train_labels = labels[:1500]
test_labels = labels[1500:]

# Nearest Neighbour instance
neighbour = NearestNeighbours(train_images, train_labels, 'l1')
# prediction
predictions = neighbour(test_images)
# score
logging.info('Nearest Neighbour instance')
score_fn(predictions, test_labels, to_print=False)


# K Nearest Neighbour instance
knn = KNearestNeighbours(train_images, train_labels, k=6, norm='l1')
# prediction
predictions = knn(test_images)
# score
logging.info('K Nearest Neighbour instance')
score_fn(predictions, test_labels, to_print=False)
