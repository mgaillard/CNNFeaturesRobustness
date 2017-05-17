#include <algorithm>
#include <queue>

#include "FeaturesIndex.h"

FeaturesIndex::FeaturesIndex() {

}

FeaturesIndex::FeaturesIndex(const vector<CnnFeatures> &features) :
        features_(features) {

}

unsigned long FeaturesIndex::size() const {
    return features_.size();
}

void FeaturesIndex::add(const CnnFeatures &feature) {
    features_.push_back(feature);
}

void FeaturesIndex::add(const vector<CnnFeatures> &features) {
    features_.insert(features_.end(), features.begin(), features.end());
}

const vector<CnnFeatures>& FeaturesIndex::features() const {
    return features_;
}

vector<pair<float, unsigned long> > FeaturesIndex::search_radius(const CnnFeatures &query_features,
                                                                 const float threshold) const {
    vector<pair<float, unsigned long> > results;

    #pragma omp parallel for shared(results)
    for (unsigned long i = 0; i < features_.size(); i++) {
        float dist = CnnFeaturesDistanceSq(query_features, features_[i]);

        if (dist <= threshold) {
            #pragma omp critical(results_update)
            {
                results.push_back(make_pair(dist, i));
            }
        }
    }

    sort(results.begin(), results.end());

    return results;
}

vector<pair<float, unsigned long> > FeaturesIndex::search_knearest(const CnnFeatures &query_features,
                                                                   const unsigned long k) const {
    priority_queue<pair<float, unsigned long> > k_nearest;

    for (unsigned long i = 0; i < features_.size(); i++) {
        float dist = CnnFeaturesDistanceSq(query_features, features_[i]);

        // There is not enough elements in the queue, we add a new element.
        if (k_nearest.size() < k) {
            k_nearest.push(make_pair(dist, i));
        } else if (k_nearest.size() >= k && k_nearest.top().first > dist) {
            k_nearest.push(make_pair(dist, i));
            k_nearest.pop();
        }
    }

    vector<pair<float, unsigned long> > results;

    while (!k_nearest.empty()) {
        results.push_back(k_nearest.top());
        k_nearest.pop();
    }
    reverse(results.begin(), results.end());

    return results;
}