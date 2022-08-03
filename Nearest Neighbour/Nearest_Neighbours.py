
import numpy as np

class NearestNeighbours:

    def __init__(self, images, labels):
        self.train_images = images.reshape(images.shape[0],-1)
        self.train_labels = labels

    def __call__(self, images):
        num_images = images.shape[0]
        test_images = images.reshape(num_images,-1)
        preds = np.zeros(num_images, dtype=np.int32)

        # find nearest training image for each of the test-images
        # using L1-distance (sum of absolute differences)
        for i in range(num_images):
            L1 = np.sum(np.abs(self.train_images - test_images[i,:]), axis=1)
            least_dist_id = np.argmin(L1)
            preds[i]=self.train_labels[least_dist_id]
        
        return preds
