import numpy as np
import cv2
import os


def load_image_dataset(self,
                       data_directory_path,
                       n=0,
                       verbose=True,
                       target_size=(128, 128),
                       interpolation=cv2.INTER_AREA):
    """Load the dataset from given directory
    """
    X_data = []
    Y_data = []
    individual_dict = {}
    category_dict = {}
    for c in os.listdir(data_directory_path):
        category_dict[c] = [n, 0]
        class_path = os.path.join(data_directory_path, c)
        for individual in os.listdir(class_path):
            individual_dict[n] = (c, individual)
            individual_images = []
            individual_path = os.path.join(class_path, individual)
            for filename in os.listdir(individual_path):
                image_path = os.path.join(individual_path, filename)
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_image_resized = cv2.resize(rgb_image,
                                               target_size,
                                               interpolation=interpolation)
                individual_images.append(rgb_image_resized)
                Y_data.append(n)
            try:
                X_data.append(np.stack(individual_images))
            except ValueError as e:
                print("Exception occured: {}".format(e))
            n += 1
            category_dict[c][1] = n - 1

    Y_data = np.vstack(Y_data)
    X_data = np.stack(X_data)

    X_data = X_data.astype("float32")
    X_data /= 255
    return X_data, Y_data, category_dict
