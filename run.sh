#!/bin/bash

# Run the program to compute the distance distributions
cd features_distances_release
./CNNFeaturesDistances > ../results/distributions.dat
cd ..
# Plot results
cd results
gnuplot < ../plot/distributions.pg
cd ..

# Run the program to benchmark the features
cd features_benchmark_release
for t in "base" "blur" "gray" "resize50" "compress10" "rotate5" "crop10"; do
    ./CNNFeaturesBenchmark --features_base ../features/features_base.h5 --features_modified "../features/features_$t.h5" > "../results/benchmark_$t.dat"
    gnuplot -e "filename='../results/benchmark_$t.dat';name='CNN Features ($t)'" "../plot/threshold.pg" > "../results/benchmark_$t.png"
done
cd ..
