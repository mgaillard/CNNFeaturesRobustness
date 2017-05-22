#!/bin/bash

source config.sh

# Run the program to compute the distance distributions
cd features_distances_release
for model in ${cnn_model[@]}; do
    ./CNNFeaturesDistances --features_directory "../features/$model" --distance $cnn_features_distance > "../results/distributions_$model.dat"
done
cd ..
# Plot results
cd results
for model in ${cnn_model[@]}; do
    gnuplot -e "filename='distributions_$model.dat'" "../plot/distributions.pg" > "distributions_$model.png"
done
cd ..

# Run the program to benchmark the features
cd features_benchmark_release
for model in ${cnn_model[@]}; do
    for t in ${transformations[@]}; do
        ./CNNFeaturesBenchmark single --features_base "../features/$model/features_base.h5" --features_modified "../features/$model/features_$t.h5" --distance $cnn_features_distance --threshold_start ${cnn_model_threshold_start[$model]} --threshold_end ${cnn_model_threshold_end[$model]} --threshold_step ${cnn_model_threshold_step[$model]} > "../results/benchmark_${model}_${t}.dat"
        gnuplot -e "filename='../results/benchmark_${model}_${t}.dat';name='$model Features ($t)'" "../plot/threshold.pg" > "../results/benchmark_${model}_${t}.png"
    done
    ./CNNFeaturesBenchmark all --features_directory "../features/$model" --distance $cnn_features_distance --threshold_start ${cnn_model_threshold_start[$model]} --threshold_end ${cnn_model_threshold_end[$model]} --threshold_step ${cnn_model_threshold_step[$model]} > "../results/benchmark_${model}_all.dat"
    gnuplot -e "filename='../results/benchmark_${model}_all.dat';name='$model Features (all)'" "../plot/threshold.pg" > "../results/benchmark_${model}_all.png"
done
cd ..
