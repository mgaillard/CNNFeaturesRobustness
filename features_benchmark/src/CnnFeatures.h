#ifndef CNNFEATURESBENCHMARK_CNNFEATURES_H
#define CNNFEATURESBENCHMARK_CNNFEATURES_H

#include <vector>
#include <string>
#include <functional>

using namespace std;

typedef vector<float> CnnFeatures;

typedef function<float(const CnnFeatures&, const CnnFeatures&)> CnnFeaturesDistanceFunction;

const string CNN_DIST_EUCLIDEAN = "euclidean";

const string CNN_DIST_EUCLIDEAN_SQUARE = "euclidean_square";

CnnFeaturesDistanceFunction MakeCnnFeaturesDistanceFunction(const string& distance_type);

float CnnFeaturesEuclideanDistanceSq(const CnnFeatures &features_a, const CnnFeatures &features_b);

float CnnFeaturesEuclideanDistance(const CnnFeatures &features_a, const CnnFeatures &features_b);

#endif //CNNFEATURESBENCHMARK_CNNFEATURES_H
