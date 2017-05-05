#!/bin/bash

# Run the program to compute the distance distributions
cd features_distances_release
./features_distances > ../results/distributions.dat
cd ..
# Plot results
cd results
gnuplot < ../plot/distributions.pg
cd ..
# Clean
rm -r features_distances_release
