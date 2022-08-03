
import logging
import numpy as np


# util
from loading_data import load_data
from Nearest_Neighbours import NearestNeighbours


logging.basicConfig(level=logging.INFO)


# download the data
images, labels = load_data()

# train-test split
train_images = images[:1500]
test_images = images[1500:]
train_labels = labels[:1500]
test_labels = labels[1500:]

# Nearest Neighbour instance
neighbour = NearestNeighbours(train_images, train_labels)
# prediction
predictions = neighbour(test_images)
# accuracy
accuracy = np.sum(predictions == test_labels) / len(predictions)
print('Model accuracy', accuracy)

# correct prediction
t = np.sum(predictions == test_labels)
f = len(predictions) - t

print('Correct prediction ', t)
print('Incorrect prediction ', f)


logging.info(f'Correct prediction : {t}')
logging.info(f'Incorrect prediction : {f}')
logging.info(f'Model accuracy : {accuracy}')
