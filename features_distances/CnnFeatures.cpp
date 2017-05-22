#include <cmath>

#include "CnnFeatures.h"

CnnFeaturesDistanceFunction MakeCnnFeaturesDistanceFunction(const string& distance_type) {
    if (distance_type.compare(CNN_DIST_EUCLIDEAN) == 0) {
        return CnnFeaturesEuclideanDistance;
    } else if (distance_type.compare(CNN_DIST_EUCLIDEAN_SQUARE) == 0) {
        return CnnFeaturesEuclideanDistanceSq;
    } else if (distance_type.compare(CNN_DIST_COSINE) == 0) {
        return CnnFeaturesCosineDistance;
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

float CnnFeaturesCosineDistance(const CnnFeatures &features_a, const CnnFeatures &features_b) {
    float dot_product = 0.0;
    float norm_a_sq = 0.0;
    float norm_b_sq = 0.0;

    for (unsigned long i = 0; i < features_a.size(); i++) {
        dot_product += features_a[i] * features_b[i];
        norm_a_sq += features_a[i] * features_a[i];
        norm_b_sq += features_b[i] * features_b[i];
    }

    return 1 - (dot_product / (sqrt(norm_a_sq) * sqrt(norm_b_sq)));
}