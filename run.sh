#!/bin/bash

source config.sh

# Run the program to compute the distance distributions
cd features_distances_release
for model in ${cnn_model[@]}; do
    for distance in ${cnn_features_distances[@]}; do
        model_distance="${model}_${distance}"
        result_directory="../results/${model_distance}"
        result_dat_file="$result_directory/distributions_${model_distance}.dat"
        result_png_file="$result_directory/distributions_${model_distance}.png"

        trace "Distribution of distances: $model_distance"
        ./CNNFeaturesDistances --features_directory "../features/$model" --distance $distance > $result_dat_file
        gnuplot -e "filename='$result_dat_file'" "../plot/distributions.pg" > $result_png_file

        # Compute the thresholds for the benchmark. The end threshold is the median of the distances between non similar images.
        cnn_model_threshold_start[$model_distance]=0
        # The end threshold is the median of the distances between the non similar images. If the median is less than 1, the end threshold is fixed at 1.
        cnn_model_threshold_end[$model_distance]=$(awk '{if($1=="non-similar"){if(int($5) < 1){print 1}else{print int($5)}}}' $result_dat_file)
        cnn_model_threshold_step[$model_distance]=$(bc <<< "scale=3;${cnn_model_threshold_end[$model_distance]} / 100")

        echo "threshold_start=${cnn_model_threshold_start[$model_distance]}"
        echo "threshold_end=${cnn_model_threshold_end[$model_distance]}"
        echo "threshold_step=${cnn_model_threshold_step[$model_distance]}"
    done
done
cd ..

# Run the program to benchmark the features
cd features_benchmark_release

# Reset the txt result file
result_txt_file="../results/results.txt"
echo "Results of CNNFeaturesBenchmark" > $result_txt_file
echo -e "Model\tDistance\tTransformation\tThreshold\tPrecision\tRecall\tF1-measure" >> $result_txt_file

# Run the benchmark
for model in ${cnn_model[@]}; do
    for distance in ${cnn_features_distances[@]}; do
        model_distance="${model}_${distance}"
        result_directory="../results/${model_distance}"

        # Benchmark on all modifications at the same time
        if [ "$multiple_transformation_benchmark" = true ] ; then
            result_dat_file="$result_directory/benchmark_${model_distance}_all.dat"
            result_png_file="$result_directory/benchmark_${model_distance}_all.png"

            trace "Benchmark all: $model_distance"
            ./CNNFeaturesBenchmark features_all $result_dat_file --directory "../features/$model" --distance $distance --threshold_start ${cnn_model_threshold_start[$model_distance]} --threshold_end ${cnn_model_threshold_end[$model_distance]} --threshold_step ${cnn_model_threshold_step[$model_distance]}
            # Plot the results
            gnuplot -e "filename='$result_dat_file';name='$model Features (all) $distance'" "../plot/threshold.pg" > $result_png_file
            # Add the best f1-measure line to the result
            echo -ne "${model}\t${distance}\tall\t" >> $result_txt_file
            awk -v max=0 '{if($4>max){line=$0; max=$4}}END{print line}' $result_dat_file >> $result_txt_file
        fi

        # Benchmark on individual transformations
        if [ "$individual_transformation_benchmark" = true ] ; then
            for t in ${transformations[@]}; do
                result_dat_file="$result_directory/benchmark_${model_distance}_${t}.dat"
                result_png_file="$result_directory/benchmark_${model_distance}_${t}.png"

                trace "Benchmark single: ${model_distance}_${t}"
                ./CNNFeaturesBenchmark features_single $result_dat_file --base "../features/$model/features_base.h5" --modified "../features/$model/features_$t.h5" --distance $distance --threshold_start ${cnn_model_threshold_start[$model_distance]} --threshold_end ${cnn_model_threshold_end[$model_distance]} --threshold_step ${cnn_model_threshold_step[$model_distance]}
                # Plot the results
                gnuplot -e "filename='$result_dat_file';name='$model Features ($t) $distance'" "../plot/threshold.pg" > $result_png_file
                # Add the best f1-measure line to the result
                echo -ne "${model}\t${distance}\t${t}\t" >> $result_txt_file
                awk -v max=0 '{if($4>max){line=$0; max=$4}}END{print line}' $result_dat_file >> $result_txt_file
            done
        fi
    done
done
cd ..
