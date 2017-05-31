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
    const unsigned long n = features_a.size();
    const float* a = features_a.data();
    const float* b = features_b.data();

    float distance_sq = 0;

    #pragma omp simd reduction(+:distance_sq)
    for (unsigned long i = 0; i < n; i++) {
        distance_sq += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return distance_sq;
}

float CnnFeaturesEuclideanDistance(const CnnFeatures &features_a, const CnnFeatures &features_b) {
    return sqrt(CnnFeaturesEuclideanDistanceSq(features_a, features_b));
}

float CnnFeaturesCosineDistance(const CnnFeatures &features_a, const CnnFeatures &features_b) {
    const unsigned long n = features_a.size();
    const float* a = features_a.data();
    const float* b = features_b.data();

    float dot_product = 0.0;
    float norm_a_sq = 0.0;
    float norm_b_sq = 0.0;

    #pragma omp simd reduction(+:dot_product, norm_a_sq, norm_b_sq)
    for (unsigned long i = 0; i < n; i++) {
        dot_product += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    return 1 - (dot_product / (sqrt(norm_a_sq) * sqrt(norm_b_sq)));
}