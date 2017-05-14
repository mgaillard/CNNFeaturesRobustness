#ifndef CNNFEATURESBENCHMARK_FEATURESINDEX_H
#define CNNFEATURESBENCHMARK_FEATURESINDEX_H

#include <vector>

#include "CnnFeatures.h"

using namespace std;

class FeaturesIndex {
public:
    FeaturesIndex();

    FeaturesIndex(const vector<CnnFeatures> &features);

    void Add(const CnnFeatures &feature);

    void Add(const vector<CnnFeatures> &features);

    vector<pair<float, unsigned long> > SearchRadius(const CnnFeatures &query_features, const float threshold) const;

    vector<pair<float, unsigned long> > SearchKNearest(const CnnFeatures &query_features, const unsigned long k) const;

private:
    vector<CnnFeatures> features_;

};


#endif //CNNFEATURESBENCHMARK_FEATURESINDEX_H
