#ifndef CNNFEATURESDISTANCES_CNNFEATURES_H
#define CNNFEATURESDISTANCES_CNNFEATURES_H

#include <vector>
#include <string>
#include <functional>

using namespace std;

typedef vector<float> CnnFeatures;

typedef function<float(const CnnFeatures&, const CnnFeatures&)> CnnFeaturesDistanceFunction;

const string CNN_DIST_EUCLIDEAN = "euclidean";

const string CNN_DIST_EUCLIDEAN_SQUARE = "euclidean_square";

const string CNN_DIST_COSINE = "cosine";

CnnFeaturesDistanceFunction MakeCnnFeaturesDistanceFunction(const string& distance_type);

float CnnFeaturesEuclideanDistanceSq(const CnnFeatures &features_a, const CnnFeatures &features_b);

float CnnFeaturesEuclideanDistance(const CnnFeatures &features_a, const CnnFeatures &features_b);

float CnnFeaturesCosineDistance(const CnnFeatures &features_a, const CnnFeatures &features_b);

#endif //CNNFEATURESDISTANCES_CNNFEATURES_H
