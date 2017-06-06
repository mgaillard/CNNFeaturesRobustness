#ifndef CNNFEATURESBENCHMARK_BENCHMARK_H
#define CNNFEATURESBENCHMARK_BENCHMARK_H

#include <vector>

#include "CnnFeatures.h"
#include "FeaturesIndex.h"
#include "BenchmarkStats.h"

using namespace std;

class Benchmark {
public:
    static vector<BenchmarkStats> single_modification(const FeaturesIndex &index,
                                                      const vector<CnnFeatures> &features_modified,
                                                      const vector<float> &thresholds);

    static vector<BenchmarkStats> all_modifications(const FeaturesIndex &index,
                                                    unsigned long nb_relevants,
                                                    const vector<float> &thresholds);

    static vector<float> generate_thresholds(float start, float end, float step);

    static void display_features_information(const vector<CnnFeatures>& features);

    static void save_stats(const vector<BenchmarkStats> &stats, const string &filename);
};

#endif //CNNFEATURESBENCHMARK_BENCHMARK_H
