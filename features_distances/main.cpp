#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/extended_p_square.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include "CnnFeatures.h"
#include "FeaturesHdf5IO.h"

using namespace std;
using namespace boost::accumulators;
namespace po = boost::program_options;

void compute_stats_similar(const CnnFeaturesDistanceFunction distance_function,
                           const vector<CnnFeatures> &base_features,
                           const vector<CnnFeatures> &similar_features) {
    boost::array<double, 3> quantile_probs = {0.25, 0.5, 0.75};

    accumulator_set<double, stats<tag::min,
            tag::max,
            tag::extended_p_square> > acc_similar(tag::extended_p_square::probabilities = quantile_probs);

    for (unsigned int i = 0; i < base_features.size(); i++) {
        float d_similar = distance_function(base_features[i], similar_features[i]);
        // Push distance to the accumulator.
        acc_similar(d_similar);
    }

    cout << extract_result<tag::min>(acc_similar) << '\t'
         << extract_result<tag::extended_p_square>(acc_similar)[0] << '\t'
         << extract_result<tag::extended_p_square>(acc_similar)[1] << '\t'
         << extract_result<tag::extended_p_square>(acc_similar)[2] << '\t'
         << extract_result<tag::max>(acc_similar) << endl;
}

void compute_stats_nonsimilar(const CnnFeaturesDistanceFunction distance_function,
                              const vector<CnnFeatures> &features) {
    boost::array<double, 3> quantile_probs = {0.25, 0.5, 0.75};

    accumulator_set<double, stats<tag::min,
            tag::max,
            tag::extended_p_square> > acc_nonsimilar(tag::extended_p_square::probabilities = quantile_probs);

    for (unsigned int i = 0; i < features.size(); i++) {
        for (unsigned int j = i + 1; j < features.size(); j++) {
            float d_nonsimilar = distance_function(features[i], features[j]);

            acc_nonsimilar(d_nonsimilar);
        }
    }

    cout << extract_result<tag::min>(acc_nonsimilar) << '\t'
         << extract_result<tag::extended_p_square>(acc_nonsimilar)[0] << '\t'
         << extract_result<tag::extended_p_square>(acc_nonsimilar)[1] << '\t'
         << extract_result<tag::extended_p_square>(acc_nonsimilar)[2] << '\t'
         << extract_result<tag::max>(acc_nonsimilar) << endl;
}

int main(int argc, const char *argv[]) {
    try {
        string directory;
        string distance_type;

        // Declare the supported command line options.
        po::options_description options_desc("./CNNFeaturesBenchmark [options]\nAllowed options:");
        options_desc.add_options()
                ("help,h", "Display a help message")
                ("features_directory,f", po::value<string>(&directory)->required(), "Path to directory of features")
                ("distance,d", po::value<string>(&distance_type)->default_value(CNN_DIST_EUCLIDEAN), "Type of distance between features");

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                          .options(options_desc)
                          .run(), vm);

        if (vm.count("help")) {
            cout << options_desc << endl;
            return EXIT_SUCCESS;
        }

        po::notify(vm);

        if (vm.count("features_directory")) {
            const CnnFeaturesDistanceFunction distance_function = MakeCnnFeaturesDistanceFunction(distance_type);

            vector<CnnFeatures> features_base = FeaturesHdf5IO::load(directory + "/features_base.h5");
            vector<CnnFeatures> features_blur = FeaturesHdf5IO::load(directory + "/features_blur.h5");
            vector<CnnFeatures> features_gray = FeaturesHdf5IO::load(directory + "/features_gray.h5");
            vector<CnnFeatures> features_resize50 = FeaturesHdf5IO::load(directory + "/features_resize50.h5");
            vector<CnnFeatures> features_compress10 = FeaturesHdf5IO::load(directory + "/features_compress10.h5");
            vector<CnnFeatures> features_rotate5 = FeaturesHdf5IO::load(directory + "/features_rotate5.h5");
            vector<CnnFeatures> features_crop10 = FeaturesHdf5IO::load(directory + "/features_crop10.h5");

            cout << "#transformation\tid\tmin\tfirst quartile\tmediane\tlast quartile\tmax" << endl;
            cout << "blur\t1\t";
            compute_stats_similar(distance_function, features_base, features_blur);

            cout << "gray\t2\t";
            compute_stats_similar(distance_function, features_base, features_gray);

            cout << "resize50\t3\t";
            compute_stats_similar(distance_function, features_base, features_resize50);

            cout << "compress10\t4\t";
            compute_stats_similar(distance_function, features_base, features_compress10);

            cout << "rotate5\t5\t";
            compute_stats_similar(distance_function, features_base, features_rotate5);

            cout << "crop10\t6\t";
            compute_stats_similar(distance_function, features_base, features_crop10);

            cout << "non-similar\t7\t";
            compute_stats_nonsimilar(distance_function, features_base);
        }
    }
    catch (exception &e) {
        cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}