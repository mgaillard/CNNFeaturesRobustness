#include <cmath>

#include "CnnFeatures.h"

CnnFeaturesDistanceFunction MakeCnnFeaturesDistanceFunction(const string& distance_type) {
    if (distance_type.compare(CNN_DIST_EUCLIDEAN) == 0) {
        return CnnFeaturesEuclideanDistance;
    } else if (distance_type.compare(CNN_DIST_EUCLIDEAN_SQUARE) == 0) {
        return CnnFeaturesEuclideanDistanceSq;
    }

    // By default, the distance is Euclidean.
    return CnnFeaturesEuclideanDistance;
}

float CnnFeaturesEuclideanDistanceSq(const CnnFeatures &features_a, const CnnFeatures &features_b) {
    float distance_sq = 0;

    for (unsigned long i = 0; i < features_a.size(); i++) {
        distance_sq += (features_a[i] - features_b[i]) * (features_a[i] - features_b[i]);
    }

    return distance_sq;
}

float CnnFeaturesEuclideanDistance(const CnnFeatures &features_a, const CnnFeatures &features_b) {
    return sqrt(CnnFeaturesEuclideanDistanceSq(features_a, features_b));
}