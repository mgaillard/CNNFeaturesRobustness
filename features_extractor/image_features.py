"""
A class that extract features from images in a directory.
"""
from os import listdir
from os.path import isfile, join

from keras.preprocessing import image
from keras.models import Model
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
import keras.applications.vgg16 as app_vgg16
import keras.applications.vgg19 as app_vgg19

import numpy as np

from sklearn.preprocessing import normalize

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


class VGG16Extractor:
    """
    Create models and preprocess images with a VGG16 neural network.
    """

    """
    Create a model based on the VGG16 neural network.
    """
    @staticmethod
    def create_model(model_type):
        if model_type == 'VGG16_predictions':
            return app_vgg16.VGG16(weights='imagenet')
        elif model_type == 'VGG16_fc2':
            base_model = app_vgg16.VGG16(weights='imagenet')
            return Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        elif model_type == 'VGG16_fc1':
            base_model = app_vgg16.VGG16(weights='imagenet')
            return Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        elif model_type == 'VGG16_flatten':
            base_model = app_vgg16.VGG16(weights='imagenet')
            return Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
        elif model_type == 'VGG16_block5_pool_avg':
            return app_vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')
        elif model_type == 'VGG16_block5_pool_max':
            return app_vgg16.VGG16(weights='imagenet', include_top=False, pooling='max')
        elif model_type == 'VGG16_block4_pool_avg':
            base_model = app_vgg16.VGG16(weights='imagenet', include_top=False)
            # add a global spatial average pooling layer
            averaged_model = base_model.get_layer('block4_pool').output
            averaged_model = GlobalAveragePooling2D()(averaged_model)
            return Model(inputs=base_model.input, outputs=averaged_model)
        elif model_type == 'VGG16_block4_pool_max':
            base_model = app_vgg16.VGG16(weights='imagenet', include_top=False)
            # add a global spatial average pooling layer
            maximized_model = base_model.get_layer('block4_pool').output
            maximized_model = GlobalMaxPooling2D()(maximized_model)
            return Model(inputs=base_model.input, outputs=maximized_model)
        elif model_type == 'VGG16_block3_pool_avg':
            base_model = app_vgg16.VGG16(weights='imagenet', include_top=False)
            # add a global spatial average pooling layer
            averaged_model = base_model.get_layer('block3_pool').output
            averaged_model = GlobalAveragePooling2D()(averaged_model)
            return Model(inputs=base_model.input, outputs=averaged_model)
        elif model_type == 'VGG16_block3_pool_max':
            base_model = app_vgg16.VGG16(weights='imagenet', include_top=False)
            # add a global spatial average pooling layer
            maximized_model = base_model.get_layer('block3_pool').output
            maximized_model = GlobalMaxPooling2D()(maximized_model)
            return Model(inputs=base_model.input, outputs=maximized_model)
        else:
            raise ValueError('The model type for the FeatureExtractor doesn\'t exist')
    
    """
    Load an image for the VGG16 neural network.
    """
    @staticmethod
    def load_image(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return app_vgg16.preprocess_input(img_array)


class VGG19Extractor:
    """
    Create models and preprocess images with a VGG19 neural network.
    """

    """
    Create a model based on the VGG19 neural network.
    """
    @staticmethod
    def create_model(model_type):
        if model_type == 'VGG19_predictions':
            return app_vgg19.VGG19(weights='imagenet')
        elif model_type == 'VGG19_fc2':
            base_model = app_vgg19.VGG19(weights='imagenet')
            return Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        elif model_type == 'VGG19_fc1':
            base_model = app_vgg19.VGG19(weights='imagenet')
            return Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        elif model_type == 'VGG19_flatten':
            base_model = app_vgg19.VGG19(weights='imagenet')
            return Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
        elif model_type == 'VGG19_block5_pool_avg':
            return app_vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')
        elif model_type == 'VGG19_block5_pool_max':
            return app_vgg19.VGG19(weights='imagenet', include_top=False, pooling='max')
        elif model_type == 'VGG19_block4_pool_avg':
            base_model = app_vgg19.VGG19(weights='imagenet', include_top=False)
            # add a global spatial average pooling layer
            averaged_model = base_model.get_layer('block4_pool').output
            averaged_model = GlobalAveragePooling2D()(averaged_model)
            return Model(inputs=base_model.input, outputs=averaged_model)
        elif model_type == 'VGG19_block4_pool_max':
            base_model = app_vgg19.VGG19(weights='imagenet', include_top=False)
            # add a global spatial average pooling layer
            maximized_model = base_model.get_layer('block4_pool').output
            maximized_model = GlobalMaxPooling2D()(maximized_model)
            return Model(inputs=base_model.input, outputs=maximized_model)
        elif model_type == 'VGG19_block3_pool_avg':
            base_model = app_vgg19.VGG19(weights='imagenet', include_top=False)
            # add a global spatial average pooling layer
            averaged_model = base_model.get_layer('block3_pool').output
            averaged_model = GlobalAveragePooling2D()(averaged_model)
            return Model(inputs=base_model.input, outputs=averaged_model)
        elif model_type == 'VGG19_block3_pool_max':
            base_model = app_vgg19.VGG19(weights='imagenet', include_top=False)
            # add a global spatial average pooling layer
            maximized_model = base_model.get_layer('block3_pool').output
            maximized_model = GlobalMaxPooling2D()(maximized_model)
            return Model(inputs=base_model.input, outputs=maximized_model)
        else:
            raise ValueError('The model type for the FeatureExtractor doesn\'t exist')
    
    """
    Load an image for the VGG19 neural network.
    """
    @staticmethod
    def load_image(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return app_vgg19.preprocess_input(img_array)


class FeatureExtractor:
    """
    Extract image features
    """

    def __init__(self, model_type):
        # Neural network to extract features
        if model_type.startswith("VGG16"):
            self.extractor = VGG16Extractor()
        elif model_type.startswith("VGG19"):
            self.extractor = VGG19Extractor()
        else:
            raise ValueError('The neural network in the model type for the FeatureExtractor doesn\'t exist')
        
        # Application of a normalization
        if model_type.endswith('_norm_l2'):
            self.normalization = 'l2'
            # Erase the norm in the model.
            model_type = model_type[:-8]
        else:
            self.normalization = 'None'
        
        # Create the model
        self.cnn_model = self.extractor.create_model(model_type)


    @staticmethod
    def list_images_directory(dir_path):
        files = []
        for filename in listdir(dir_path):
            path = join(dir_path, filename)
            if isfile(path) and path.lower().endswith(('.jpg', '.png')):
                files.append(filename)
        files.sort()
        return files


    def load_images(self, path, image_files):
        images = []
        for img in image_files:
            images.append(self.extractor.load_image(join(path, img)))
        return np.concatenate(images)


    def extract(self, path):
        features = Features()

        features.directory_path = path
        features.images = self.list_images_directory(path)

        images = self.load_images(path, features.images)
        features.features = self.cnn_model.predict(images)

        if self.normalization in ['l1', 'l2', 'max']:
            features.features = normalize(features.features, norm=self.normalization, axis=1)

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
