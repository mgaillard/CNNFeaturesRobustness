#include <iostream>
#include <vector>
#include <boost/program_options.hpp>

#include "CnnFeatures.h"
#include "FeaturesHdf5IO.h"
#include "FeaturesIndex.h"
#include "Benchmark.h"

using namespace std;
namespace po = boost::program_options;

void BenchmarkModification(const string &features_base_path, const string &features_modified_path);

int main(int argc, const char *argv[]) {
    try {
        string features_base_path;
        string features_modified_path;
        // Declare the supported command line options.
        po::options_description options_desc("./CNNFeaturesBenchmark [options]\nAllowed options:");
        options_desc.add_options()
                ("help,h", "Display a help message")
                ("features_base,b", po::value<string>(&features_base_path)->required(), "Path to base features")
                ("features_modified,m", po::value<string>(&features_modified_path)->required(), "Path to modified features");

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);

        if (vm.count("help")) {
            cout << options_desc << endl;
            return EXIT_SUCCESS;
        }

        po::notify(vm);

        BenchmarkModification(features_base_path, features_modified_path);
    }
    catch (exception &e) {
        cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void BenchmarkModification(const string &features_base_path, const string &features_modified_path) {
    vector<CnnFeatures> features_base = FeaturesHdf5IO::load(features_base_path);
    vector<CnnFeatures> features_modified = FeaturesHdf5IO::load(features_modified_path);

    FeaturesIndex index(features_base);

    vector<float> thresholds;

    for (int i = 0; i < 100;i++) {
        thresholds.push_back(100 + 100*i);
    }

    vector<BenchmarkStats> stats = Benchmark::Modification(index, features_modified, thresholds);

    for (BenchmarkStats& stat : stats) {
        cout << stat.threshold() << "\t"
             << stat.mean_precision() << "\t"
             << stat.mean_recall() << "\t"
             << stat.mean_f1measure() << endl;
    }
}
