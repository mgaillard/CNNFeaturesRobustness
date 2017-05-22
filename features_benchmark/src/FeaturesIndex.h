#ifndef CNNFEATURESBENCHMARK_FEATURESINDEX_H
#define CNNFEATURESBENCHMARK_FEATURESINDEX_H

#include <vector>

#include "CnnFeatures.h"

using namespace std;

class FeaturesIndex {
public:
    FeaturesIndex(const CnnFeaturesDistanceFunction& cnn_features_distance);

    FeaturesIndex(const CnnFeaturesDistanceFunction& cnn_features_distance, 
                  const vector<CnnFeatures> &features);

    unsigned long size() const;

    void add(const CnnFeatures &feature);

    void add(const vector<CnnFeatures> &features);

    const vector<CnnFeatures>& features() const;

    vector<pair<float, unsigned long> > search_radius(const CnnFeatures &query_features, const float threshold) const;

    vector<pair<float, unsigned long> > search_knearest(const CnnFeatures &query_features, const unsigned long k) const;

private:
    const CnnFeaturesDistanceFunction cnn_features_distance_;

    vector<CnnFeatures> features_;
};


#endif //CNNFEATURESBENCHMARK_FEATURESINDEX_H
