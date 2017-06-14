"""
A program that extract features from images using CNN.
"""
import argparse

import sys

from os.path import isdir

import numpy as np

from image_features import FeatureExtractor, FeaturesNpzIO, FeaturesHdf5IO

def main():
    # parse program arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('directory',
                        type=str,
                        help='Which directory to process')

    parser.add_argument('--output',
                        type=str,
                        help='In which file to save the features',
                        default='dataset')

    parser.add_argument('--model',
                        type=str,
                        choices=['ResNet50_predictions',
                                 'ResNet50_flatten_1',
                                 'ResNet50_flatten_1_norm_l2',
                                 'ResNet50_avg_pool_avg',
                                 'ResNet50_avg_pool_avg_norm_l2',
                                 'ResNet50_avg_pool_max',
                                 'ResNet50_avg_pool_max_norm_l2',
                                 'ResNet50_activation_46_avg',
                                 'ResNet50_activation_46_max',
                                 'ResNet50_activation_43_avg',
                                 'ResNet50_activation_43_max',
                                 'ResNet50_activation_40_avg',
                                 'ResNet50_activation_40_max',
                                 'ResNet50_activation_37_avg',
                                 'ResNet50_activation_37_max',
                                 'ResNet50_activation_34_avg',
                                 'ResNet50_activation_34_max',
                                 'ResNet50_activation_31_avg',
                                 'ResNet50_activation_31_max',
                                 'ResNet50_activation_28_avg',
                                 'ResNet50_activation_28_max',
                                 'VGG16_predictions',
                                 'VGG16_fc2',
                                 'VGG16_fc2_norm_l2',
                                 'VGG16_fc1',
                                 'VGG16_fc1_norm_l2',
                                 'VGG16_flatten',
                                 'VGG16_flatten_norm_l2',
                                 'VGG16_block5_pool_avg',
                                 'VGG16_block5_pool_avg_norm_l2',
                                 'VGG16_block5_pool_max',
                                 'VGG16_block5_pool_max_norm_l2',
                                 'VGG16_block4_pool_avg',
                                 'VGG16_block4_pool_avg_norm_l2',
                                 'VGG16_block4_pool_max',
                                 'VGG16_block4_pool_max_norm_l2',
                                 'VGG16_block3_pool_avg',
                                 'VGG16_block3_pool_avg_norm_l2',
                                 'VGG16_block3_pool_max',
                                 'VGG16_block3_pool_max_norm_l2',
                                 'VGG19_predictions',
                                 'VGG19_fc2',
                                 'VGG19_fc2_norm_l2',
                                 'VGG19_fc1',
                                 'VGG19_fc1_norm_l2',
                                 'VGG19_flatten',
                                 'VGG19_flatten_norm_l2',
                                 'VGG19_block5_pool_avg',
                                 'VGG19_block5_pool_avg_norm_l2',
                                 'VGG19_block5_pool_max',
                                 'VGG19_block5_pool_max_norm_l2',
                                 'VGG19_block4_pool_avg',
                                 'VGG19_block4_pool_avg_norm_l2',
                                 'VGG19_block4_pool_max',
                                 'VGG19_block4_pool_max_norm_l2',
                                 'VGG19_block3_pool_avg',
                                 'VGG19_block3_pool_avg_norm_l2',
                                 'VGG19_block3_pool_max',
                                 'VGG19_block3_pool_max_norm_l2'],
                        help='Which model to use to extract features',
                        default='VGG16_block5_pool_avg')

    parser.add_argument('--format',
                        type=str,
                        choices=['npz', 'h5'],
                        help='In which format to save the features',
                        default='npz')

    args = parser.parse_args()

    if not isdir(args.directory):
        print('The provided directory doesn\'t exist.')
        sys.exit()
    
    # Extract features
    extractor = FeatureExtractor(args.model)
    features = extractor.extract(args.directory)

    # Display information about the features
    print('Number of images: {}'.format(features.features.shape[0]))
    print('Features shape: {}'.format(features.features.shape))
    print('Mean of the features of the first image: {}'.format(np.mean(features.features[0])))
    print('L2 norm of the features of the first image: {}'.format(np.linalg.norm(features.features[0], 2)))
    print('Features of the first image:\n{}'.format(features.features[0]))

    # Save the features
    if args.format == 'npz':
        FeaturesNpzIO.save(args.output, features)
    elif args.format == 'h5':
        FeaturesHdf5IO.save(args.output, features)


if __name__ == "__main__":
    main()
