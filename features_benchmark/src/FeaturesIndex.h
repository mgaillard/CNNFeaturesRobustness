#ifndef CNNFEATURESBENCHMARK_FEATURESINDEX_H
#define CNNFEATURESBENCHMARK_FEATURESINDEX_H

#include <vector>

#include "CnnFeatures.h"

using namespace std;

class FeaturesIndex {
public:
    FeaturesIndex();

    FeaturesIndex(const vector<CnnFeatures> &features);

    unsigned long size() const;

    void add(const CnnFeatures &feature);

    void add(const vector<CnnFeatures> &features);

    const vector<CnnFeatures>& features() const;

    vector<pair<float, unsigned long> > search_radius(const CnnFeatures &query_features, const float threshold) const;

    vector<pair<float, unsigned long> > search_knearest(const CnnFeatures &query_features, const unsigned long k) const;

private:
    vector<CnnFeatures> features_;

};


#endif //CNNFEATURESBENCHMARK_FEATURESINDEX_H
