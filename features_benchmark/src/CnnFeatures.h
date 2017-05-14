#ifndef CNNFEATURESBENCHMARK_CNNFEATURES_H
#define CNNFEATURESBENCHMARK_CNNFEATURES_H

#include <vector>

using namespace std;

typedef vector<float> CnnFeatures;

float CnnFeaturesDistanceSq(const CnnFeatures &features_a, const CnnFeatures &features_b);

#endif //CNNFEATURESBENCHMARK_CNNFEATURES_H
