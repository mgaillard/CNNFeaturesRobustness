CNNFeaturesRobustness
==================
Test the robustness of image features from Convolutional Neural Networks against modifications. This software is developed as part of a master thesis between INSA Lyon, France and Universität Passau, Deutschland.


How it works ?
--------------
We analyze the distribution of distances between features of transformed images and base images. As a reference, we also analyze the distribution of distances between non similar images. The features are extracted from various layers of various neural networks and are compared using a Euclidean distance.

Dependencies
------------
- Keras / Tensorflow
- Boost
- HDF5
- ImageMagick
- gnuplot

Usage
-----

```bash
# Change the config of $image_dir to point to the image directories
$ nano config.sh
# Compile and extract features
$ bash prepare.sh
# Compute and display the distance distributions
$ bash run.sh
# Results are in the results directory
```

License
-------
See the LICENSE file.
