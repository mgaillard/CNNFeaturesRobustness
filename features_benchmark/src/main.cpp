#include <iostream>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "CnnFeatures.h"
#include "FeaturesHdf5IO.h"
#include "Benchmark.h"

using namespace std;
namespace ba = boost::algorithm;
namespace po = boost::program_options;

vector<string> ListFeatureFiles(const string &dir_path);

void BenchmarkSingleModificationFeatures(const string &features_base_path,
                                         const string &features_modified_path,
                                         const string &distance_type,
                                         float threshold_start,
                                         float threshold_end,
                                         float threshold_step,
                                         const string &output);

void BenchmarkSingleModificationCodes(const string &codes_base_path,
                                      const string &codes_modified_path,
                                      const string &distance_type,
                                      float threshold_start,
                                      float threshold_end,
                                      float threshold_step,
                                      const string &output);

void BenchmarkAllModificationsFeatures(const string &features_directory_path,
                                       const string &distance_type,
                                       float threshold_start,
                                       float threshold_end,
                                       float threshold_step,
                                       const string &output);

void BenchmarkAllModificationsCodes(const string &codes_directory_path,
                                    const string &distance_type,
                                    float threshold_start,
                                    float threshold_end,
                                    float threshold_step,
                                    const string &output);

int main(int argc, const char *argv[]) {
    try {
        string benchmark_type;
        string output;
        string distance_type;
        float threshold_start;
        float threshold_end;
        float threshold_step;

        // Declare the supported command line options.
        po::options_description options_desc("./CNNFeaturesBenchmark [options]\nAllowed options:");
        options_desc.add_options()
                ("help,h", "Display a help message")
                ("benchmark_type", po::value<string>(&benchmark_type)->required(),
                     "Set the type of benchmark to execute: 'features_single' or 'features_all' or 'codes_single' or 'codes_all'")
                ("output", po::value<string>(&output)->required(), "Set the output of the benchmark")
                ("directory,d", po::value<string>(), "Path to directory of features/codes")
                ("base,b", po::value<string>(), "Path to base features/codes")
                ("modified,m", po::value<string>(), "Path to modified features/codes")
                ("distance,d", po::value<string>(&distance_type)->required(),
                      "Type of distance between features/codes")
                ("threshold_start", po::value<float>(&threshold_start)->default_value(0.0f), "Starting threshold")
                ("threshold_end", po::value<float>(&threshold_end)->default_value(1.0f), "Ending threshold")
                ("threshold_step", po::value<float>(&threshold_step)->default_value(0.02f), "Threshold step");

        po::positional_options_description pos_options_desc;
        pos_options_desc.add("benchmark_type", 1);
        pos_options_desc.add("output", 2);

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

        if (vm.count("benchmark_type") && vm.count("output")) {
            if (benchmark_type.compare("features_single") == 0) {
                if (vm.count("base") && vm.count("modified")) {
                    string features_base_path = vm["base"].as<string>();
                    string features_modified_path = vm["modified"].as<string>();

                    BenchmarkSingleModificationFeatures(features_base_path,
                                                        features_modified_path,
                                                        distance_type,
                                                        threshold_start,
                                                        threshold_end,
                                                        threshold_step,
                                                        output);
                } else {
                    cerr
                            << "Error: the --base and --modified options are required for the benchmark 'features_single'."
                            << endl;
                }

            } else if (benchmark_type.compare("codes_single") == 0) {
                if (vm.count("base") && vm.count("modified")) {
                    string codes_base_path = vm["base"].as<string>();
                    string codes_modified_path = vm["modified"].as<string>();

                    BenchmarkSingleModificationCodes(codes_base_path,
                                                     codes_modified_path,
                                                     distance_type,
                                                     threshold_start,
                                                     threshold_end,
                                                     threshold_step,
                                                     output);
                } else {
                    cerr
                            << "Error: the --base and --modified options are required for the benchmark 'codes_single'."
                            << endl;
                }

            } else if (benchmark_type.compare("features_all") == 0) {
                if (vm.count("directory")) {
                    string features_directory_path = vm["directory"].as<string>();

                    BenchmarkAllModificationsFeatures(features_directory_path,
                                                      distance_type,
                                                      threshold_start,
                                                      threshold_end,
                                                      threshold_step,
                                                      output);
                } else {
                    cerr
                            << "Error: the --directory option is required for the benchmark 'features_all'."
                            << endl;
                }
            } else if (benchmark_type.compare("codes_all") == 0) {
                if (vm.count("directory")) {
                    string codes_directory_path = vm["directory"].as<string>();

                    BenchmarkAllModificationsCodes(codes_directory_path,
                                                   distance_type,
                                                   threshold_start,
                                                   threshold_end,
                                                   threshold_step,
                                                   output);
                } else {
                    cerr
                            << "Error: the --directory option is required for the benchmark 'codes_all'."
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

    sort(files.begin(), files.end());

    return files;
}

void BenchmarkSingleModificationFeatures(const string &features_base_path,
                                         const string &features_modified_path,
                                         const string &distance_type,
                                         float threshold_start,
                                         float threshold_end,
                                         float threshold_step,
                                         const string &output) {
    vector<CnnFeatures> features_base = FeaturesHdf5IO::loadFeatures(features_base_path);
    vector<CnnFeatures> features_modified = FeaturesHdf5IO::loadFeatures(features_modified_path);

    CnnFeaturesDistanceFunction distance_function = MakeCnnFeaturesDistanceFunction(distance_type);
    Index<CnnFeatures> index(distance_function);
    index.add(features_base);

    vector<float> thresholds = Benchmark<CnnFeatures>::generate_thresholds(threshold_start, threshold_end, threshold_step);

    vector<BenchmarkStats> stats = Benchmark<CnnFeatures>::single_modification(index, features_modified, thresholds);

    Benchmark<CnnFeatures>::save_stats(stats, output);
}

void BenchmarkSingleModificationCodes(const string &codes_base_path,
                                      const string &codes_modified_path,
                                      const string &distance_type,
                                      float threshold_start,
                                      float threshold_end,
                                      float threshold_step,
                                      const string &output) {
    vector<CnnCode> codes_base = FeaturesHdf5IO::loadBinaryCodes(codes_base_path);
    vector<CnnCode> codes_modified = FeaturesHdf5IO::loadBinaryCodes(codes_modified_path);

    CnnCodeDistanceFunction distance_function = MakeCnnCodesDistanceFunction(distance_type);
    Index<CnnCode> index(distance_function);
    index.add(codes_base);

    vector<float> thresholds = Benchmark<CnnCode>::generate_thresholds(threshold_start, threshold_end, threshold_step);

    vector<BenchmarkStats> stats = Benchmark<CnnCode>::single_modification(index, codes_modified, thresholds);

    Benchmark<CnnCode>::save_stats(stats, output);
}

void BenchmarkAllModificationsFeatures(const string &features_directory_path,
                                       const string &distance_type,
                                       float threshold_start,
                                       float threshold_end,
                                       float threshold_step,
                                       const string &output) {
    vector<string> feature_files = ListFeatureFiles(features_directory_path);
    unsigned long nb_relevant = feature_files.size();

    // Load all features
    CnnFeaturesDistanceFunction distance_function = MakeCnnFeaturesDistanceFunction(distance_type);
    Index<CnnFeatures> index(distance_function);
    for (const string &file : feature_files) {
        vector<CnnFeatures> features = FeaturesHdf5IO::loadFeatures(file);
        index.add(features);
    }

    vector<float> thresholds = Benchmark<CnnFeatures>::generate_thresholds(threshold_start, threshold_end, threshold_step);

    vector<BenchmarkStats> stats = Benchmark<CnnFeatures>::all_modifications(index, nb_relevant, thresholds);

    Benchmark<CnnFeatures>::save_stats(stats, output);
}

void BenchmarkAllModificationsCodes(const string &codes_directory_path,
                                    const string &distance_type,
                                    float threshold_start,
                                    float threshold_end,
                                    float threshold_step,
                                    const string &output) {
    vector<string> codes_files = ListFeatureFiles(codes_directory_path);
    unsigned long nb_relevant = codes_files.size();

    // Load all codes
    CnnCodeDistanceFunction distance_function = MakeCnnCodesDistanceFunction(distance_type);
    Index<CnnCode> index(distance_function);
    for (const string &file : codes_files) {
        vector<CnnCode> codes = FeaturesHdf5IO::loadBinaryCodes(file);
        index.add(codes);
    }

    vector<float> thresholds = Benchmark<CnnCode>::generate_thresholds(threshold_start, threshold_end, threshold_step);

    vector<BenchmarkStats> stats = Benchmark<CnnCode>::all_modifications(index, nb_relevant, thresholds);

    Benchmark<CnnCode>::save_stats(stats, output);
}