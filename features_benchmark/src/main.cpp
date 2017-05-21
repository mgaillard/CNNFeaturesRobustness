#include <iostream>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "CnnFeatures.h"
#include "FeaturesHdf5IO.h"
#include "FeaturesIndex.h"
#include "Benchmark.h"

using namespace std;
namespace ba = boost::algorithm;
namespace po = boost::program_options;

vector<string> ListFeatureFiles(const string &dir_path);

void BenchmarkSingleModification(const string &features_base_path,
                                 const string &features_modified_path,
                                 float threshold_start,
                                 float threshold_end,
                                 float threshold_step);

void BenchmarkAllModifications(const string &features_directory_path,
                               float threshold_start,
                               float threshold_end,
                               float threshold_step);

int main(int argc, const char *argv[]) {
    try {
        string benchmark_type;
        float threshold_start;
        float threshold_end;
        float threshold_step;

        // Declare the supported command line options.
        po::options_description options_desc("./CNNFeaturesBenchmark [options]\nAllowed options:");
        options_desc.add_options()
                ("help,h", "Display a help message")
                ("benchmark_type", po::value<string>(&benchmark_type)->required(),
                 "Set the type of benchmark to execute: 'single' or 'all'")
                ("features_directory,d", po::value<string>(), "Path to directory of features")
                ("features_base,b", po::value<string>(), "Path to base features")
                ("features_modified,m", po::value<string>(), "Path to modified features")
                ("threshold_start", po::value<float>(&threshold_start)->default_value(100.0), "Starting threshold")
                ("threshold_end", po::value<float>(&threshold_end)->default_value(10000.0), "Ending threshold")
                ("threshold_step", po::value<float>(&threshold_step)->default_value(100.0), "Threshold step");

        po::positional_options_description pos_options_desc;
        pos_options_desc.add("benchmark_type", 1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                          .options(options_desc)
                          .positional(pos_options_desc)
                          .run(), vm);

        if (vm.count("help")) {
            cout << options_desc << endl;
            return EXIT_SUCCESS;
        }

        po::notify(vm);

        if (vm.count("benchmark_type")) {
            if (benchmark_type.compare("single") == 0) {
                if (vm.count("features_base") && vm.count("features_modified")) {
                    string features_base_path = vm["features_base"].as<string>();
                    string features_modified_path = vm["features_modified"].as<string>();

                    BenchmarkSingleModification(features_base_path,
                                                features_modified_path,
                                                threshold_start,
                                                threshold_end,
                                                threshold_step);
                } else {
                    cerr
                            << "Error: the --features_base and --features_modified options are required for the benchmark 'single'."
                            << endl;
                }

            } else if (benchmark_type.compare("all") == 0) {
                if (vm.count("features_directory")) {
                    string features_directory_path = vm["features_directory"].as<string>();

                    BenchmarkAllModifications(features_directory_path,
                                              threshold_start,
                                              threshold_end,
                                              threshold_step);
                } else {
                    cerr
                            << "Error: the --features_base option is required for the benchmark 'all'."
                            << endl;
                }
            }
        }
    }
    catch (exception &e) {
        cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

vector<string> ListFeatureFiles(const string &dir_path) {
    vector<string> files;

    DIR *dir = opendir(dir_path.c_str());
    if (dir != NULL) {
        struct dirent *dir_entry;
        while ((dir_entry = readdir(dir)) != NULL) {
            if (strcmp(dir_entry->d_name, ".") && strcmp(dir_entry->d_name, "..")) {
                string file_path;
                if (dir_path.back() == '/') {
                    file_path = dir_path + dir_entry->d_name;
                } else {
                    file_path = dir_path + '/' + dir_entry->d_name;
                }

                // If it's a regular file.
                struct stat file_info;
                if (stat(file_path.c_str(), &file_info) == 0
                    && S_ISREG(file_info.st_mode)
                    && ba::ends_with(file_path, ".h5")) {
                    files.push_back(file_path);
                }
            }
        }

        closedir(dir);
    } else {
        cout << "Error reading directory." << endl;
    }

    return files;
}

void BenchmarkSingleModification(const string &features_base_path,
                                 const string &features_modified_path,
                                 float threshold_start,
                                 float threshold_end,
                                 float threshold_step) {
    vector<CnnFeatures> features_base = FeaturesHdf5IO::load(features_base_path);
    vector<CnnFeatures> features_modified = FeaturesHdf5IO::load(features_modified_path);

    FeaturesIndex index(features_base);

    vector<float> thresholds = Benchmark::generate_thresholds(threshold_start, threshold_end, threshold_step);

    vector<BenchmarkStats> stats = Benchmark::single_modification(index, features_modified, thresholds);

    Benchmark::display_stats(stats);
}

void BenchmarkAllModifications(const string &features_directory_path,
                               float threshold_start,
                               float threshold_end,
                               float threshold_step) {
    vector<string> feature_files = ListFeatureFiles(features_directory_path);
    unsigned long nb_relevant = feature_files.size();

    // Load all features
    FeaturesIndex index;
    for (const string &file : feature_files) {
        vector<CnnFeatures> features = FeaturesHdf5IO::load(file);
        index.add(features);
    }

    vector<float> thresholds = Benchmark::generate_thresholds(threshold_start, threshold_end, threshold_step);

    vector<BenchmarkStats> stats = Benchmark::all_modifications(index, nb_relevant, thresholds);

    Benchmark::display_stats(stats);
}