#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <cmath>

#include "../src/CnnFeatures.h"
#include "../src/FeaturesIndex.h"
#include "../src/Benchmark.h"
#include "../src/BenchmarkStats.h"

using namespace std;

const double EPSILON = 0.0001;

TEST_CASE("Features are indexed", "[features_index]") {
    vector<CnnFeatures> features = {
            {0, 1},
            {0, 2},
            {0, 3},
            {0, 4},
    };

    FeaturesIndex index(CnnFeaturesEuclideanDistanceSq, features);

    SECTION("Search in a radius of 1") {
        vector<pair<float, unsigned long> > results = index.search_radius({0, 5}, 1);

        REQUIRE(results.size() == 1);

        REQUIRE(results[0].first == 1);
        REQUIRE(results[0].second == 3);
    }

    SECTION("Search in a radius of 4") {
        vector<pair<float, unsigned long> > results = index.search_radius({0, 0}, 4);

        REQUIRE(results.size() == 2);

        REQUIRE(results[0].first == 1);
        REQUIRE(results[0].second == 0);

        REQUIRE(results[1].first == 4);
        REQUIRE(results[1].second == 1);
    }

    SECTION("Search the 2 nearest") {
        vector<pair<float, unsigned long> > results = index.search_knearest({0, 0}, 2);

        REQUIRE(results.size() == 2);

        REQUIRE(results[0].first == 1);
        REQUIRE(results[0].second == 0);

        REQUIRE(results[1].first == 4);
        REQUIRE(results[1].second == 1);
    }

    SECTION("Search the 2 nearest") {
        vector<pair<float, unsigned long> > results = index.search_knearest({1, 2.5f}, 2);

        REQUIRE(results.size() == 2);

        REQUIRE(results[0].first == 1.25f);
        REQUIRE(results[0].second == 1);

        REQUIRE(results[1].first == 1.25f);
        REQUIRE(results[1].second == 2);
    }

    SECTION("Search the 4 nearest") {
        vector<pair<float, unsigned long> > results = index.search_knearest({0, 0}, 4);

        REQUIRE(results.size() == 4);

        REQUIRE(results[0].first == 1);
        REQUIRE(results[0].second == 0);

        REQUIRE(results[1].first == 4);
        REQUIRE(results[1].second == 1);

        REQUIRE(results[2].first == 9);
        REQUIRE(results[2].second == 2);

        REQUIRE(results[3].first == 16);
        REQUIRE(results[3].second == 3);
    }
}

TEST_CASE("Statistics are computed", "[statistics]") {
    vector<pair<float, unsigned long> > results = {
            make_pair(1, 3),
            make_pair(2, 1),
            make_pair(3, 0),
            make_pair(4, 2),
            make_pair(5, 4),
    };

    SECTION("Relevants are 1, 3, 5") {
        BenchmarkStats stats(5);
        stats.process(results, {1, 3, 5});

        REQUIRE(abs(stats.mean_precision() - 2.0/5) < EPSILON);
        REQUIRE(abs(stats.mean_recall() - 2.0/3) < EPSILON);
        REQUIRE(abs(stats.mean_f1measure() - 1.0/2) < EPSILON);
    }

    SECTION("No relevants elements") {
        BenchmarkStats stats(5);
        stats.process(results, {});

        REQUIRE(abs(stats.mean_precision() - 0.0) < EPSILON);
        REQUIRE(abs(stats.mean_recall() - 1.0) < EPSILON);
        REQUIRE(abs(stats.mean_f1measure() - 0.0) < EPSILON);
    }
}

TEST_CASE("Statistics for different threshold are computed", "[statistics]") {
    vector<pair<float, unsigned long> > results = {
            make_pair(1, 1),
            make_pair(2, 2),
            make_pair(3, 3),
            make_pair(4, 4),
            make_pair(5, 5),
            make_pair(6, 6),
    };

    vector<unsigned long> relevants = {1, 2, 3};

    SECTION("Threshold 1") {
        BenchmarkStats stats(1);
        stats.process(results, relevants);

        REQUIRE(abs(stats.mean_precision() - 1.0) < EPSILON);
        REQUIRE(abs(stats.mean_recall() - 1.0/3.0) < EPSILON);
        REQUIRE(abs(stats.mean_f1measure() - 1.0/2) < EPSILON);
    }

    SECTION("Threshold 2") {
        BenchmarkStats stats(2);
        stats.process(results, relevants);

        REQUIRE(abs(stats.mean_precision() - 1.0) < EPSILON);
        REQUIRE(abs(stats.mean_recall() - 2.0/3.0) < EPSILON);
        REQUIRE(abs(stats.mean_f1measure() - 4.0/5.0) < EPSILON);
    }

    SECTION("Threshold 3") {
        BenchmarkStats stats(3);
        stats.process(results, relevants);

        REQUIRE(abs(stats.mean_precision() - 1.0) < EPSILON);
        REQUIRE(abs(stats.mean_recall() - 1.0) < EPSILON);
        REQUIRE(abs(stats.mean_f1measure() - 1.0) < EPSILON);
    }

    SECTION("Threshold 4") {
        BenchmarkStats stats(4);
        stats.process(results, relevants);

        REQUIRE(abs(stats.mean_precision() - 3.0/4.0) < EPSILON);
        REQUIRE(abs(stats.mean_recall() - 1.0) < EPSILON);
        REQUIRE(abs(stats.mean_f1measure() - 6.0/7.0) < EPSILON);
    }

    SECTION("Threshold 5") {
        BenchmarkStats stats(5);
        stats.process(results, relevants);

        REQUIRE(abs(stats.mean_precision() - 3.0/5.0) < EPSILON);
        REQUIRE(abs(stats.mean_recall() - 1.0) < EPSILON);
        REQUIRE(abs(stats.mean_f1measure() - 3.0/4.0) < EPSILON);
    }

    SECTION("Threshold 6") {
        BenchmarkStats stats(6);
        stats.process(results, relevants);

        REQUIRE(abs(stats.mean_precision() - 1.0/2.0) < EPSILON);
        REQUIRE(abs(stats.mean_recall() - 1.0) < EPSILON);
        REQUIRE(abs(stats.mean_f1measure() - 2.0/3.0) < EPSILON);
    }
}

TEST_CASE("Benchmark with single modification are executed", "[benchmark]") {
    vector<CnnFeatures> features_base = {
            {0, 1},
            {0, 2},
            {0, 4},
            {0, 5},
    };

    vector<CnnFeatures> features_modified = {
            {1, 1},
            {1, 2},
            {1, 4},
            {1, 5},
    };

    FeaturesIndex index(CnnFeaturesEuclideanDistanceSq, features_base);

    SECTION("Threshold: 1 and 2") {
        vector<BenchmarkStats> stats = Benchmark::single_modification(index, features_modified, {1.0, 2.0});

        REQUIRE(abs(stats[0].mean_precision() - 1.0) < EPSILON);
        REQUIRE(abs(stats[0].mean_recall() - 1.0) < EPSILON);
        REQUIRE(abs(stats[0].mean_f1measure() - 1.0) < EPSILON);

        REQUIRE(abs(stats[1].mean_precision() - 0.5) < EPSILON);
        REQUIRE(abs(stats[1].mean_recall() - 1.0) < EPSILON);
        REQUIRE(abs(stats[1].mean_f1measure() - 2.0/3) < EPSILON);
    }
}

TEST_CASE("Benchmark with all modifications are executed", "[benchmark]") {
    vector<CnnFeatures> features_a = {
            {0, 0},
            {0, 4},
            {0, 8},
            {0, 12},
    };

    vector<CnnFeatures> features_b = {
            {1, 0},
            {1, 4},
            {1, 8},
            {1, 12},
    };

    vector<CnnFeatures> features_c = {
            {2, 0},
            {2, 4},
            {2, 8},
            {2, 12},
    };

    FeaturesIndex index(CnnFeaturesEuclideanDistanceSq);
    index.add(features_a);
    index.add(features_b);
    index.add(features_c);

    SECTION("Threshold: 4") {
        vector<BenchmarkStats> stats = Benchmark::all_modifications(index, 3, {4.0});

        REQUIRE(abs(stats[0].mean_precision() - 1.0) < EPSILON);
        REQUIRE(abs(stats[0].mean_recall() - 1.0) < EPSILON);
        REQUIRE(abs(stats[0].mean_f1measure() - 1.0) < EPSILON);
    }
}

TEST_CASE("Threshold generation", "[generate_threshold]") {
    vector<float> thresholds = Benchmark::generate_thresholds(10.0, 40.0, 10.0);

    REQUIRE(thresholds.size() == 4);

    REQUIRE(abs(thresholds[0] - 10.0) < EPSILON);
    REQUIRE(abs(thresholds[1] - 20.0) < EPSILON);
    REQUIRE(abs(thresholds[2] - 30.0) < EPSILON);
    REQUIRE(abs(thresholds[3] - 40.0) < EPSILON);
}

TEST_CASE("Distances", "[distance]") {
    CnnFeatures features_a = {1.0, 5.0, 3.0, -1.2};
    CnnFeatures features_b = {6.0, 4.0, 2.0, -2.5};

    SECTION("Euclidean") {
        float dist = CnnFeaturesEuclideanDistance(features_a, features_b);

        REQUIRE(abs(dist - 5.356304) < EPSILON);
    }

    SECTION("Euclidean Square") {
        float dist = CnnFeaturesEuclideanDistanceSq(features_a, features_b);

        REQUIRE(abs(dist - 28.69) < EPSILON);
    }

    SECTION("Cosine") {
        float dist = CnnFeaturesCosineDistance(features_a, features_b);

        REQUIRE(abs(dist - 0.2651323) < EPSILON);
    }
}