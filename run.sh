#!/bin/bash

source config.sh

# Run the program to compute the distance distributions
cd features_distances_release
for model in ${cnn_model[@]}; do
    ./CNNFeaturesDistances --features_directory "../features/$model" --distance $cnn_features_distance > "../results/distributions_${model}_${cnn_features_distance}.dat"
done
cd ..
# Plot results
cd results
for model in ${cnn_model[@]}; do
    gnuplot -e "filename='distributions_${model}_${cnn_features_distance}.dat'" "../plot/distributions.pg" > "distributions_${model}_${cnn_features_distance}.png"
done
cd ..

# Run the program to benchmark the features
cd features_benchmark_release
for model in ${cnn_model[@]}; do
    model_distance="${model}_${cnn_features_distance}"

    # Benchmark on individual transformations
    if [ "$individual_transformation_benchmark" = true ] ; then
        for t in ${transformations[@]}; do
            ./CNNFeaturesBenchmark single --features_base "../features/$model/features_base.h5" --features_modified "../features/$model/features_$t.h5" --distance $cnn_features_distance --threshold_start ${cnn_model_threshold_start[$model_distance]} --threshold_end ${cnn_model_threshold_end[$model_distance]} --threshold_step ${cnn_model_threshold_step[$model_distance]} > "../results/benchmark_${model_distance}_${t}.dat"
            gnuplot -e "filename='../results/benchmark_${model_distance}_${t}.dat';name='$model Features ($t) $cnn_features_distance'" "../plot/threshold.pg" > "../results/benchmark_${model}_${cnn_features_distance}_${t}.png"
        done
    fi
    
    # Benchmark on all modifications at the same time
    if [ "$multiple_transformation_benchmark" = true ] ; then
        ./CNNFeaturesBenchmark all --features_directory "../features/$model" --distance $cnn_features_distance --threshold_start ${cnn_model_threshold_start[$model_distance]} --threshold_end ${cnn_model_threshold_end[$model_distance]} --threshold_step ${cnn_model_threshold_step[$model_distance]} > "../results/benchmark_${model_distance}_all.dat"
        gnuplot -e "filename='../results/benchmark_${model_distance}_all.dat';name='$model Features (all) $cnn_features_distance'" "../plot/threshold.pg" > "../results/benchmark_${model_distance}_all.png"
    fi
done
cd ..
