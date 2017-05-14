#ifndef CNNFEATURESBENCHMARK_BENCHMARK_H
#define CNNFEATURESBENCHMARK_BENCHMARK_H

#include <vector>

#include "CnnFeatures.h"
#include "FeaturesIndex.h"
#include "BenchmarkStats.h"

using namespace std;

class Benchmark {
public:
    static vector<BenchmarkStats> Modification(const FeaturesIndex &index,
                                               const vector<CnnFeatures> &features_modified,
                                               vector<float> thresholds);
};

#endif //CNNFEATURESBENCHMARK_BENCHMARK_H
