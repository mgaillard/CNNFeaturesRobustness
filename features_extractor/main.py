"""
A program that extract features from images using CNN.
"""
import argparse

import sys

from os.path import isdir

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
                        choices=['VGG16_block5_pool_avg',
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
                                 'VGG16_block3_pool_max_norm_l2'],
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

    extractor = FeatureExtractor(args.model)
    features = extractor.extract(args.directory)

    if args.format == 'npz':
        FeaturesNpzIO.save(args.output, features)
    elif args.format == 'h5':
        FeaturesHdf5IO.save(args.output, features)


if __name__ == "__main__":
    main()
