#include "CnnFeatures.h"

float CnnFeaturesDistanceSq(const CnnFeatures &features_a, const CnnFeatures &features_b) {
    float distance_sq = 0;

    for (unsigned long i = 0; i < features_a.size(); i++) {
        distance_sq += (features_a[i] - features_b[i]) * (features_a[i] - features_b[i]);
    }

    return distance_sq;
}