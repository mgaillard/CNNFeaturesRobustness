#include <algorithm>

#include "BenchmarkStats.h"

BenchmarkStats::BenchmarkStats(float threshold) :
        threshold_(threshold),
        total_precision_(0.0),
        total_recall_(0.0),
        total_f1measure_(0.0),
        number_results_(0) {

}

void BenchmarkStats::process(const vector<pair<float, unsigned long> > &results,
                             const vector<unsigned long> &relevants) {
    double precision, recall, f1measure;

    unsigned long relevant = relevants.size();
    unsigned long retrieved_relevant = relevant_items(results, relevants);
    unsigned long retrieved = retrieved_items(results);

    if (retrieved > 0) {
        precision = static_cast<double>(retrieved_relevant) / retrieved;
    } else {
        precision = 1.0;
    }

    if (relevant > 0) {
        recall = static_cast<double>(retrieved_relevant) / relevant;
    } else {
        recall = 1.0;
    }

    if (precision + recall > 0) {
        f1measure = 2 * precision * recall / (precision + recall);
    } else {
        f1measure = 0;
    }

    total_precision_ += precision;
    total_recall_ += recall;
    total_f1measure_ += f1measure;
    number_results_ += 1;
}

float BenchmarkStats::threshold() const {
    return threshold_;
}

double BenchmarkStats::mean_precision() const {
    return total_precision_ / number_results_;
}

double BenchmarkStats::mean_recall() const {
    return total_recall_ / number_results_;
}

double BenchmarkStats::mean_f1measure() const {
    return total_f1measure_ / number_results_;
}

unsigned long BenchmarkStats::relevant_items(const vector<pair<float, unsigned long> > &results,
                                             const vector<unsigned long> &relevants) {
    unsigned long nb_relevant = 0;

    for (const pair<float, unsigned long> &result : results) {
        if (result.first <= threshold_
        && binary_search(relevants.begin(), relevants.end(), result.second)) {
            nb_relevant++;
        }
    }

    return nb_relevant;
}

unsigned long BenchmarkStats::retrieved_items(const vector<pair<float, unsigned long> > &results) {
    unsigned long nb_retrieved = 0;

    for (const pair<float, unsigned long> &result : results) {
        if (result.first <= threshold_) {
            nb_retrieved++;
        }
    }

    return nb_retrieved;
}