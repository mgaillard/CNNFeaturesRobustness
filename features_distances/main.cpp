#include <iostream>
#include <vector>
#include <cmath>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/extended_p_square.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include "Features.h"
#include "FeaturesHdf5IO.h"

using namespace std;
using namespace boost::accumulators;

/**
 * Compute the distance between two feature vectors.
 * @param a A feature vector
 * @param b A feature vector
 * @return The distance between the two feature vectors.
 */
double features_distance(const vector<float> &a, const vector<float> &b) {
    double distance_sq = 0.0;

    for (unsigned int i = 0; i < a.size(); i++) {
        distance_sq += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return sqrt(distance_sq);
}

void compute_stats_similar(const Features &base_features, const Features &similar_features) {
    boost::array<double, 3> quantile_probs = {0.25, 0.5, 0.75};

    accumulator_set<double, stats<tag::min,
            tag::max,
            tag::extended_p_square> > acc_similar(tag::extended_p_square::probabilities = quantile_probs);

    for (unsigned int i = 0; i < base_features.nb_images(); i++) {
        float d_similar = features_distance(base_features.image_features(i), similar_features.image_features(i));
        // Push distance to the accumulator.
        acc_similar(d_similar);
    }

    cout << extract_result<tag::min>(acc_similar) << '\t'
         << extract_result<tag::extended_p_square>(acc_similar)[0] << '\t'
         << extract_result<tag::extended_p_square>(acc_similar)[1] << '\t'
         << extract_result<tag::extended_p_square>(acc_similar)[2] << '\t'
         << extract_result<tag::max>(acc_similar) << endl;
}

void compute_stats_nonsimilar(const Features &features) {
    boost::array<double, 3> quantile_probs = {0.25, 0.5, 0.75};

    accumulator_set<double, stats<tag::min,
            tag::max,
            tag::extended_p_square> > acc_nonsimilar(tag::extended_p_square::probabilities = quantile_probs);

    for (unsigned int i = 0; i < features.nb_images(); i++) {
        for (unsigned int j = i + 1; j < features.nb_images(); j++) {
            float d_nonsimilar = features_distance(features.image_features(i), features.image_features(j));

            acc_nonsimilar(d_nonsimilar);
        }
    }

    cout << extract_result<tag::min>(acc_nonsimilar) << '\t'
         << extract_result<tag::extended_p_square>(acc_nonsimilar)[0] << '\t'
         << extract_result<tag::extended_p_square>(acc_nonsimilar)[1] << '\t'
         << extract_result<tag::extended_p_square>(acc_nonsimilar)[2] << '\t'
         << extract_result<tag::max>(acc_nonsimilar) << endl;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "The program takes one argument: the path to the directory of features" << endl;
        return 1;
    }
    
    string directory = argv[1];

    Features features_base = FeaturesHdf5IO::load(directory + "/features_base.h5");
    Features features_blur = FeaturesHdf5IO::load(directory + "/features_blur.h5");
    Features features_gray = FeaturesHdf5IO::load(directory + "/features_gray.h5");
    Features features_resize50 = FeaturesHdf5IO::load(directory + "/features_resize50.h5");
    Features features_compress10 = FeaturesHdf5IO::load(directory + "/features_compress10.h5");
    Features features_rotate5 = FeaturesHdf5IO::load(directory + "/features_rotate5.h5");
    Features features_crop10 = FeaturesHdf5IO::load(directory + "/features_crop10.h5");

    cout << "#transformation\tid\tmin\tfirst quartile\tmediane\tlast quartile\tmax" << endl;
    cout << "blur\t1\t";
    compute_stats_similar(features_base, features_blur);

    cout << "gray\t2\t";
    compute_stats_similar(features_base, features_gray);

    cout << "resize50\t3\t";
    compute_stats_similar(features_base, features_resize50);

    cout << "compress10\t4\t";
    compute_stats_similar(features_base, features_compress10);

    cout << "rotate5\t5\t";
    compute_stats_similar(features_base, features_rotate5);

    cout << "crop10\t6\t";
    compute_stats_similar(features_base, features_crop10);

    cout << "non-similar\t7\t";
    compute_stats_nonsimilar(features_base);

    return 0;
}