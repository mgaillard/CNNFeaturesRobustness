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
import keras.applications.resnet50 as app_resnet50
import keras.applications.xception as app_xception
import keras.applications.inception_v3 as app_inception_v3

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

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
            return VGG16Extractor.extract_layer(base_model, 'fc2')
        elif model_type == 'VGG16_fc1':
            base_model = app_vgg16.VGG16(weights='imagenet')
            return VGG16Extractor.extract_layer(base_model, 'fc1')
        elif model_type == 'VGG16_flatten':
            base_model = app_vgg16.VGG16(weights='imagenet')
            return VGG16Extractor.extract_layer(base_model, 'flatten')
        elif model_type == 'VGG16_block5_pool_avg':
            return app_vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')
        elif model_type == 'VGG16_block5_pool_max':
            return app_vgg16.VGG16(weights='imagenet', include_top=False, pooling='max')
        elif model_type == 'VGG16_block4_pool_avg':
            base_model = app_vgg16.VGG16(weights='imagenet', include_top=False)
            return VGG16Extractor.extract_layer(base_model, 'block4_pool', 'avg')
        elif model_type == 'VGG16_block4_pool_max':
            base_model = app_vgg16.VGG16(weights='imagenet', include_top=False)
            return VGG16Extractor.extract_layer(base_model, 'block4_pool', 'max')
        elif model_type == 'VGG16_block3_pool_avg':
            base_model = app_vgg16.VGG16(weights='imagenet', include_top=False)
            return VGG16Extractor.extract_layer(base_model, 'block3_pool', 'avg')
        elif model_type == 'VGG16_block3_pool_max':
            base_model = app_vgg16.VGG16(weights='imagenet', include_top=False)
            return VGG16Extractor.extract_layer(base_model, 'block3_pool', 'max')
        else:
            raise ValueError('The model type for the FeatureExtractor doesn\'t exist')


    @staticmethod
    def extract_layer(base_model, layer, pooling=None):
        # add a global spatial average pooling layer
        model = base_model.get_layer(layer).output
        if pooling == 'avg':
            model = GlobalAveragePooling2D()(model)
        elif pooling == 'max':
            model = GlobalMaxPooling2D()(model)
        return Model(inputs=base_model.input, outputs=model)

   
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
            return VGG19Extractor.extract_layer(base_model, 'fc2')
        elif model_type == 'VGG19_fc1':
            base_model = app_vgg19.VGG19(weights='imagenet')
            return VGG19Extractor.extract_layer(base_model, 'fc1')
        elif model_type == 'VGG19_flatten':
            base_model = app_vgg19.VGG19(weights='imagenet')
            return VGG19Extractor.extract_layer(base_model, 'flatten')
        elif model_type == 'VGG19_block5_pool_avg':
            return app_vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')
        elif model_type == 'VGG19_block5_pool_max':
            return app_vgg19.VGG19(weights='imagenet', include_top=False, pooling='max')
        elif model_type == 'VGG19_block4_pool_avg':
            base_model = app_vgg19.VGG19(weights='imagenet', include_top=False)
            return VGG19Extractor.extract_layer(base_model, 'block4_pool', 'avg')
        elif model_type == 'VGG19_block4_pool_max':
            base_model = app_vgg19.VGG19(weights='imagenet', include_top=False)
            return VGG19Extractor.extract_layer(base_model, 'block4_pool', 'max')
        elif model_type == 'VGG19_block3_pool_avg':
            base_model = app_vgg19.VGG19(weights='imagenet', include_top=False)
            return VGG19Extractor.extract_layer(base_model, 'block3_pool', 'avg')
        elif model_type == 'VGG19_block3_pool_max':
            base_model = app_vgg19.VGG19(weights='imagenet', include_top=False)
            return VGG19Extractor.extract_layer(base_model, 'block3_pool', 'max')
        else:
            raise ValueError('The model type for the FeatureExtractor doesn\'t exist')


    @staticmethod
    def extract_layer(base_model, layer, pooling=None):
        # add a global spatial average pooling layer
        model = base_model.get_layer(layer).output
        if pooling == 'avg':
            model = GlobalAveragePooling2D()(model)
        elif pooling == 'max':
            model = GlobalMaxPooling2D()(model)
        return Model(inputs=base_model.input, outputs=model)


    """
    Load an image for the VGG19 neural network.
    """
    @staticmethod
    def load_image(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return app_vgg19.preprocess_input(img_array)


class ResNet50Extractor:
    """
    Create models and preprocess images with a ResNet50 neural network.
    """

    """
    Create a model based on the ResNet50 neural network.
    """
    @staticmethod
    def create_model(model_type):
        if model_type == 'ResNet50_predictions':
            return app_resnet50.ResNet50(weights='imagenet')
        elif model_type == 'ResNet50_flatten_1':
            base_model = app_resnet50.ResNet50(weights='imagenet')
            return ResNet50Extractor.extract_layer(base_model, 'flatten_1')
        elif model_type == 'ResNet50_avg_pool_avg':
            return app_resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        elif model_type == 'ResNet50_avg_pool_max':
            return app_resnet50.ResNet50(weights='imagenet', include_top=False, pooling='max')
        elif model_type == 'ResNet50_activation_46_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_46', 'avg')
        elif model_type == 'ResNet50_activation_46_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_46', 'max')
        elif model_type == 'ResNet50_activation_43_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_43', 'avg')
        elif model_type == 'ResNet50_activation_43_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_43', 'max')
        elif model_type == 'ResNet50_activation_40_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_40', 'avg')
        elif model_type == 'ResNet50_activation_40_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_40', 'max')
        elif model_type == 'ResNet50_activation_37_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_37', 'avg')
        elif model_type == 'ResNet50_activation_37_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_37', 'max')
        elif model_type == 'ResNet50_activation_34_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_34', 'avg')
        elif model_type == 'ResNet50_activation_34_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_34', 'max')
        elif model_type == 'ResNet50_activation_31_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_31', 'avg')
        elif model_type == 'ResNet50_activation_31_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_31', 'max')
        elif model_type == 'ResNet50_activation_28_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_28', 'avg')
        elif model_type == 'ResNet50_activation_28_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_28', 'max')
        elif model_type == 'ResNet50_activation_25_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_25', 'avg')
        elif model_type == 'ResNet50_activation_25_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_25', 'max')
        elif model_type == 'ResNet50_activation_22_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_22', 'avg')
        elif model_type == 'ResNet50_activation_22_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_22', 'max')
        elif model_type == 'ResNet50_activation_19_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_19', 'avg')
        elif model_type == 'ResNet50_activation_19_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_19', 'max')
        elif model_type == 'ResNet50_activation_16_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_16', 'avg')
        elif model_type == 'ResNet50_activation_16_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_16', 'max')
        elif model_type == 'ResNet50_activation_13_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_13', 'avg')
        elif model_type == 'ResNet50_activation_13_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_13', 'max')
        elif model_type == 'ResNet50_activation_10_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_10', 'avg')
        elif model_type == 'ResNet50_activation_10_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_10', 'max')
        elif model_type == 'ResNet50_activation_7_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_7', 'avg')
        elif model_type == 'ResNet50_activation_7_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_7', 'max')
        elif model_type == 'ResNet50_activation_4_avg':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_4', 'avg')
        elif model_type == 'ResNet50_activation_4_max':
            base_model = app_resnet50.ResNet50(weights='imagenet', include_top=False)
            return ResNet50Extractor.extract_layer(base_model, 'activation_4', 'max')
        else:
            raise ValueError('The model type for the FeatureExtractor doesn\'t exist')


    @staticmethod
    def extract_layer(base_model, layer, pooling=None):
        # add a global spatial average pooling layer
        model = base_model.get_layer(layer).output
        if pooling == 'avg':
            model = GlobalAveragePooling2D()(model)
        elif pooling == 'max':
            model = GlobalMaxPooling2D()(model)
        return Model(inputs=base_model.input, outputs=model)


    """
    Load an image for the ResNet50 neural network.
    """
    @staticmethod
    def load_image(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return app_resnet50.preprocess_input(img_array)

class XceptionExtractor:
    """
    Create models and preprocess images with a Xception neural network.
    """

    """
    Create a model based on the Xception neural network.
    """
    @staticmethod
    def create_model(model_type):
        if model_type == 'Xception_predictions':
            return app_xception.Xception(weights='imagenet')
        elif model_type == 'Xception_block14_sepconv2_act_avg':
            return app_xception.Xception(weights='imagenet', include_top=False, pooling='avg')
        elif model_type == 'Xception_block14_sepconv2_act_max':
            return app_xception.Xception(weights='imagenet', include_top=False, pooling='max')
        elif model_type == 'Xception_block14_sepconv1_act_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'block14_sepconv1_act', 'avg')
        elif model_type == 'Xception_block14_sepconv1_act_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'block14_sepconv1_act', 'max')
        elif model_type == 'Xception_add_12_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_12', 'avg')
        elif model_type == 'Xception_add_12_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_12', 'max')
        elif model_type == 'Xception_add_11_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_11', 'avg')
        elif model_type == 'Xception_add_11_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_11', 'max')
        elif model_type == 'Xception_add_10_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_10', 'avg')
        elif model_type == 'Xception_add_10_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_10', 'max')
        elif model_type == 'Xception_add_9_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_9', 'avg')
        elif model_type == 'Xception_add_9_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_9', 'max')
        elif model_type == 'Xception_add_8_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_8', 'avg')
        elif model_type == 'Xception_add_8_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_8', 'max')
        elif model_type == 'Xception_add_7_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_7', 'avg')
        elif model_type == 'Xception_add_7_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_7', 'max')
        elif model_type == 'Xception_add_6_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_6', 'avg')
        elif model_type == 'Xception_add_6_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_6', 'max')
        elif model_type == 'Xception_add_5_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_5', 'avg')
        elif model_type == 'Xception_add_5_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_5', 'max')
        elif model_type == 'Xception_add_4_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_4', 'avg')
        elif model_type == 'Xception_add_4_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_4', 'max')
        elif model_type == 'Xception_add_3_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_3', 'avg')
        elif model_type == 'Xception_add_3_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_3', 'max')
        elif model_type == 'Xception_add_2_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_2', 'avg')
        elif model_type == 'Xception_add_2_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_2', 'max')
        elif model_type == 'Xception_add_1_avg':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_1', 'avg')
        elif model_type == 'Xception_add_1_max':
            base_model = app_xception.Xception(weights='imagenet', include_top=False)
            return XceptionExtractor.extract_layer(base_model, 'add_1', 'max')
        else:
            raise ValueError('The model type for the FeatureExtractor doesn\'t exist')


    @staticmethod
    def extract_layer(base_model, layer, pooling=None):
        # add a global spatial average pooling layer
        model = base_model.get_layer(layer).output
        if pooling == 'avg':
            model = GlobalAveragePooling2D()(model)
        elif pooling == 'max':
            model = GlobalMaxPooling2D()(model)
        return Model(inputs=base_model.input, outputs=model)


    """
    Load an image for the Xception neural network.
    """
    @staticmethod
    def load_image(image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return app_xception.preprocess_input(img_array)


class InceptionV3Extractor:
    """
    Create models and preprocess images with a InceptionV3 neural network.
    """

    """
    Create a model based on the InceptionV3 neural network.
    """
    @staticmethod
    def create_model(model_type):
        if model_type == 'InceptionV3_predictions':
            return app_inception_v3.InceptionV3(weights='imagenet')
        elif model_type == 'InceptionV3_mixed10_avg':
            return app_inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        elif model_type == 'InceptionV3_mixed10_max':
            return app_inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='max')
        elif model_type == 'InceptionV3_mixed9_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed9', 'avg')
        elif model_type == 'InceptionV3_mixed9_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed9', 'max')
        elif model_type == 'InceptionV3_mixed8_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed8', 'avg')
        elif model_type == 'InceptionV3_mixed8_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed8', 'max')
        elif model_type == 'InceptionV3_mixed7_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed7', 'avg')
        elif model_type == 'InceptionV3_mixed7_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed7', 'max')
        elif model_type == 'InceptionV3_mixed6_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed6', 'avg')
        elif model_type == 'InceptionV3_mixed6_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed6', 'max')
        elif model_type == 'InceptionV3_mixed5_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed5', 'avg')
        elif model_type == 'InceptionV3_mixed5_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed5', 'max')
        elif model_type == 'InceptionV3_mixed4_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed4', 'avg')
        elif model_type == 'InceptionV3_mixed4_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed4', 'max')
        elif model_type == 'InceptionV3_mixed3_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed3', 'avg')
        elif model_type == 'InceptionV3_mixed3_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed3', 'max')
        elif model_type == 'InceptionV3_mixed2_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed2', 'avg')
        elif model_type == 'InceptionV3_mixed2_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed2', 'max')
        elif model_type == 'InceptionV3_mixed1_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed1', 'avg')
        elif model_type == 'InceptionV3_mixed1_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed1', 'max')
        elif model_type == 'InceptionV3_mixed0_avg':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed0', 'avg')
        elif model_type == 'InceptionV3_mixed0_max':
            base_model = app_inception_v3.InceptionV3(weights='imagenet', include_top=False)
            return InceptionV3Extractor.extract_layer(base_model, 'mixed0', 'max')
        else:
            raise ValueError('The model type for the FeatureExtractor doesn\'t exist')


    @staticmethod
    def extract_layer(base_model, layer, pooling=None):
        # add a global spatial average pooling layer
        model = base_model.get_layer(layer).output
        if pooling == 'avg':
            model = GlobalAveragePooling2D()(model)
        elif pooling == 'max':
            model = GlobalMaxPooling2D()(model)
        return Model(inputs=base_model.input, outputs=model)


    """
    Load an image for the InceptionV3 neural network.
    """
    @staticmethod
    def load_image(image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return app_inception_v3.preprocess_input(img_array)


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
        elif model_type.startswith("ResNet50"):
            self.extractor = ResNet50Extractor()
        elif model_type.startswith("Xception"):
            self.extractor = XceptionExtractor()
        elif model_type.startswith("InceptionV3"):
            self.extractor = InceptionV3Extractor()
        else:
            raise ValueError('The neural network in the model type for the FeatureExtractor doesn\'t exist')
        
        # Application of a normalization
        if model_type.endswith('_norm_l2'):
            self.normalization = 'l2'
            # Erase the norm in the model.
            model_type = model_type[:-8]
        else:
            self.normalization = 'None'

        # Application of a dimensionality reduction
        if model_type.endswith('_pca_64'):
            # In the case of a PCA, the number of images should be greater than the number of components kept
            self.reduction = 'pca_64'
            # Erase the norm in the model.
            model_type = model_type[:-7]
        else:
            self.reduction = 'None'
        
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

        if self.reduction == 'pca_64':
            pca = PCA(n_components=64)
            features.features = pca.fit_transform(features.features)

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
