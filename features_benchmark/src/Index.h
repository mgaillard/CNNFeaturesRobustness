#ifndef CNNFEATURESBENCHMARK_INDEX_H
#define CNNFEATURESBENCHMARK_INDEX_H

#include <vector>
#include <algorithm>
#include <queue>
#include <functional>
#include "CnnFeatures.h"
#include "CnnCode.h"

using namespace std;

template<typename T>
class Index {
public:
    typedef T EntryType;
    typedef function<float(const T &, const T &)> DistanceFunction;

    Index(const DistanceFunction distance_function) : distance_function_(distance_function) {

    }

    Index(const DistanceFunction distance_function, const vector<EntryType> &entries) :
            distance_function_(distance_function),
            entries_(entries) {

    }

    unsigned long size() const {
        return entries_.size();
    }

    void add(const EntryType &entry) {
        entries_.push_back(entry);
    }

    void add(const vector<EntryType> &entry) {
        entries_.insert(entries_.end(), entry.begin(), entry.end());
    }

    const vector<EntryType> &features() const {
        return entries_;
    }

    vector<pair<float, unsigned long> > search_radius(const EntryType &query, const float threshold) const {
        vector<pair<float, unsigned long> > results;

        #pragma omp parallel for shared(results)
        for (unsigned long i = 0; i < entries_.size(); i++) {
            float dist = distance_function_(query, entries_[i]);

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

    vector<pair<float, unsigned long> > search_knearest(const EntryType &query, const unsigned long k) const {
        priority_queue<pair<float, unsigned long> > k_nearest;

        for (unsigned long i = 0; i < entries_.size(); i++) {
            float dist = distance_function_(query, entries_[i]);

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

private:
    const DistanceFunction distance_function_;

    vector<EntryType> entries_;
};


#endif //CNNFEATURESBENCHMARK_INDEX_H
