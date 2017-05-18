#ifndef CNNFEATURESBENCHMARK_BENCHMARKSTATS_H
#define CNNFEATURESBENCHMARK_BENCHMARKSTATS_H

#include <vector>

using namespace std;

class BenchmarkStats {
public:

    BenchmarkStats(float threshold);

    /**
     * Process the results of a query.
     * Compute the precision, recall and F1-measure.
     * @param results The results of a query.
     * @param relevants A sorted vector of relevant elements.
     */
    void process(const vector<pair<float, unsigned long> > &results, const vector<unsigned long> &relevants);

    float threshold() const;

    double mean_precision() const;

    double mean_recall() const;

    double mean_f1measure() const;

private:
    unsigned long relevant_items(const vector<pair<float, unsigned long> > &results,
                                 const vector<unsigned long> &relevants);

    unsigned long retrieved_items(const vector<pair<float, unsigned long> > &results);

    float threshold_;
    double total_precision_;
    double total_recall_;
    double total_f1measure_;
    unsigned long number_results_;
};


#endif //CNNFEATURESBENCHMARK_BENCHMARKSTATS_H
