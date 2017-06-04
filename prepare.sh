#!/bin/bash

source config.sh

# Result directory
mkdir results
for model in ${cnn_model[@]}; do
    for distance in ${cnn_features_distances[@]}; do
        mkdir "results/${model}_${distance}"
    done
done

# Compile the program to compute the distance distributions
mkdir features_distances_release
cd features_distances_release
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_NATIVE_ARCH=ON ../features_distances
make -j 4
cd ..

# Compile the program to benchmark the features
mkdir features_benchmark_release
cd features_benchmark_release
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_NATIVE_ARCH=ON ../features_benchmark
make -j 4
cd ..

# Transform images
mkdir "$tmp_image_dir"
mkdir "$tmp_image_dir/base"
cp -a "$image_dir/." "$tmp_image_dir/base/"

for t in ${transformations[@]}; do
    trace "Transformation: ${transformation_name[$t]}"
    mkdir "$tmp_image_dir/$t"
    eval ${transformation_command[$t]}
done

# Extract the features
mkdir features
for model in ${cnn_model[@]}; do
    mkdir "features/$model"
    for t in ${transformations[@]}; do
        trace "Extract features: $model > ${transformation_name[$t]}"
        python3 features_extractor/main.py --output "features/$model/features_$t.h5" --model $model --format h5 "$tmp_image_dir/$t"
    done
done
