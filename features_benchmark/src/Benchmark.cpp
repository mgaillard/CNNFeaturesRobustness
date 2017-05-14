#include <iostream>

#include "Benchmark.h"

using namespace std;

vector<BenchmarkStats> Benchmark::Modification(const FeaturesIndex &index,
                                               const vector<CnnFeatures> &features_modified,
                                               vector<float> thresholds) {
    float threshold_max = 0.0;
    vector<BenchmarkStats> stats;

    for (float t : thresholds) {
        stats.push_back(BenchmarkStats(t));
        threshold_max = max(threshold_max, t);
    }

    for (unsigned long i = 0; i < features_modified.size(); i++) {
        vector<pair<float, unsigned long> > results = index.SearchRadius(features_modified[i], threshold_max);

        for (BenchmarkStats& stat : stats) {
            stat.process(results, {i});
        }
    }

    return stats;
}