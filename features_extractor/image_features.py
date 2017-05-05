"""
A class that extract features from images in a directory.
"""
from os import listdir
from os.path import isfile, join

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input

import numpy as np

import h5py

class Features:
    """
    Features of images in a directory
    """

    def __init__(self):
        # Path to an image directory
        self.directory_path = ""
        # List of image files
        self.images = []
        # Features of images
        self.features = []


    def nb_images(self):
        return len(self.images)


class FeatureExtractor:
    """
    Extract image features
    """

    def __init__(self):
        # Model to extract features
        self.cnn_model = VGG16(weights='imagenet', include_top=False, pooling='avg')


    @staticmethod
    def list_images_directory(dir_path):
        files = []
        for filename in listdir(dir_path):
            path = join(dir_path, filename)
            if isfile(path) and path.lower().endswith(('.jpg', '.png')):
                files.append(filename)
        files.sort()
        return files


    @staticmethod
    def load_image(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)


    @staticmethod
    def load_images(path, image_files):
        images = []
        for img in image_files:
            images.append(FeatureExtractor.load_image(join(path, img)))
        return np.concatenate(images)


    def extract(self, path):
        features = Features()

        features.directory_path = path
        features.images = FeatureExtractor.list_images_directory(path)

        images = FeatureExtractor.load_images(path, features.images)
        features.features = self.cnn_model.predict(images)

        return features


class FeaturesNpzIO:
    """
    Provide methods to save and load features in npz files.
    """

    @staticmethod
    def save(filename, features):
        """
        Save the features in a NPZ file.
        """
        np.savez_compressed(filename,
                            directory_path=features.directory_path,
                            images=features.images,
                            features=features.features)


    @staticmethod
    def load(filename):
        """
        Load the features from a NPZ file.
        """
        features = Features()

        npzfile = np.load(filename)
        features.directory_path = npzfile['directory_path']
        features.images = npzfile['images']
        features.features = npzfile['features']

        return features


class FeaturesHdf5IO:
    """
    Provide methods to save and load features in hdf5 files.
    """

    @staticmethod
    def save(filename, features):
        """
        Save the features in a HDF5 file.
        Warning only features are saved.
        """
        hdf5_file = h5py.File(filename, 'w')
        hdf5_file.create_dataset('features', data=features.features)
        hdf5_file.close()


    @staticmethod
    def load(filename):
        """
        Load the features from a HDF5 file.
        Warning only features are loaded.
        """
        features = Features()

        hdf5_file = h5py.File(filename, 'r')
        features.features = hdf5_file['features'][:]
        hdf5_file.close()

        return features
