#ifndef CNNFEATURESBENCHMARK_BENCHMARK_H
#define CNNFEATURESBENCHMARK_BENCHMARK_H

#include <vector>
#include <iostream>
#include <fstream>

#include "CnnFeatures.h"
#include "Index.h"
#include "BenchmarkStats.h"

using namespace std;

template<typename T>
class Benchmark {
public:
    static vector<BenchmarkStats> single_modification(const Index<T> &index,
                                                      const vector<T> &entries_modified,
                                                      const vector<float> &thresholds) {
        float threshold_max = 0.0;
        vector<BenchmarkStats> stats;

        for (float t : thresholds) {
            stats.push_back(BenchmarkStats(t));
            threshold_max = max(threshold_max, t);
        }

        for (unsigned long i = 0; i < entries_modified.size(); i++) {
            vector<pair<float, unsigned long> > results = index.search_radius(entries_modified[i], threshold_max);

            #pragma omp parallel for shared(stats, results)
            for (unsigned long s = 0; s < stats.size(); s++) {
                stats[s].process(results, {i});
            }
        }

        return stats;
    }

    static vector<BenchmarkStats> all_modifications(const Index<T> &index,
                                                    unsigned long nb_relevants,
                                                    const vector<float> &thresholds) {
        float threshold_max = 0.0;
        vector<BenchmarkStats> stats;

        for (float t : thresholds) {
            stats.push_back(BenchmarkStats(t));
            threshold_max = max(threshold_max, t);
        }

        /*
         * Relevant features are the ith features of each feature files.
         * In each feature file, there is index.size() / nb_relevants features.
         * There is a total of nb_relevants feature files.
         *
         * For example, with 3 files (base and two modifications) and 4 features per file,
         * the index contains the images: 0, 1, 2, 3,
         *                                4, 5, 6, 7,
         *                                8, 9, 10, 11.
         *
         * The relevant features if the query is 0 (or 4, or 8) are: 0, 4 and 8.
         * The relevant features if the query is 1 (or 5, or 9) are: 1, 5 and 9.
         * And so on...
         *
         * To generate the list of relevant features for each feature in the index,
         * we build a base_relevants list which corresponds to the relevant features for the first query.
         * Then for the other queries, we just add a constant to the base_relevants list.
         */
        vector<unsigned long> base_relevants;
        unsigned long features_per_file = index.size() / nb_relevants;
        for (unsigned long r = 0; r < nb_relevants;r++) {
            base_relevants.push_back(r * features_per_file);
        }

        for (unsigned long i = 0; i < index.size(); i++) {
            vector<pair<float, unsigned long> > results = index.search_radius(index.features()[i], threshold_max);

            vector<unsigned long> relevants(nb_relevants);
            // Generate the indexes of relevant features for this query.
            for (unsigned long j = 0; j < relevants.size(); j++) {
                relevants[j] = base_relevants[j] + i % features_per_file;
            }

            #pragma omp parallel for shared(stats, results, relevants)
            for (unsigned long s = 0; s < stats.size(); s++) {
                stats[s].process(results, relevants);
            }
        }

        return stats;
    }

    static vector<float> generate_thresholds(float start, float end, float step) {
        vector<float> thresholds;

        for (float t = start; t <= end; t += step) {
            thresholds.push_back(t);
        }

        return thresholds;
    }

    static void save_stats(const vector<BenchmarkStats> &stats, const string &filename) {
        ofstream out_file(filename, ios::out | ios::trunc);
        if (out_file.is_open()) {
            for (const BenchmarkStats &stat : stats) {
                out_file << stat.threshold() << "\t"
                         << stat.mean_precision() << "\t"
                         << stat.mean_recall() << "\t"
                         << stat.mean_f1measure() << endl;
            }

            out_file.close();
        } else {
            cout << "Unable to open output file";
        }
    }
};

#endif //CNNFEATURESBENCHMARK_BENCHMARK_H
