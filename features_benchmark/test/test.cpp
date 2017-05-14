#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "../src/CnnFeatures.h"
#include "../src/FeaturesIndex.h"
#include "../src/Benchmark.h"
#include "../src/BenchmarkStats.h"


TEST_CASE("Features are indexed", "[features_index]") {
    vector<CnnFeatures> features = {
            {0, 1},
            {0, 2},
            {0, 3},
            {0, 4},
    };

    FeaturesIndex index(features);

    SECTION("Search in a radius of 1") {
        vector<pair<float, unsigned long> > results = index.SearchRadius({0, 5}, 1);

        REQUIRE(results.size() == 1);

        REQUIRE(results[0].first == 1);
        REQUIRE(results[0].second == 3);
    }

    SECTION("Search in a radius of 4") {
        vector<pair<float, unsigned long> > results = index.SearchRadius({0, 0}, 4);

        REQUIRE(results.size() == 2);

        REQUIRE(results[0].first == 1);
        REQUIRE(results[0].second == 0);

        REQUIRE(results[1].first == 4);
        REQUIRE(results[1].second == 1);
    }

    SECTION("Search the 2 nearest") {
        vector<pair<float, unsigned long> > results = index.SearchKNearest({0, 0}, 2);

        REQUIRE(results.size() == 2);

        REQUIRE(results[0].first == 1);
        REQUIRE(results[0].second == 0);

        REQUIRE(results[1].first == 4);
        REQUIRE(results[1].second == 1);
    }

    SECTION("Search the 2 nearest") {
        vector<pair<float, unsigned long> > results = index.SearchKNearest({1, 2.5f}, 2);

        REQUIRE(results.size() == 2);

        REQUIRE(results[0].first == 1.25f);
        REQUIRE(results[0].second == 1);

        REQUIRE(results[1].first == 1.25f);
        REQUIRE(results[1].second == 2);
    }

    SECTION("Search the 4 nearest") {
        vector<pair<float, unsigned long> > results = index.SearchKNearest({0, 0}, 4);

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

TEST_CASE("Statistics are computed", "[benchmark_stats]") {
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

        REQUIRE(stats.mean_precision() == 2.0/5);
        REQUIRE(stats.mean_recall() == 2.0/3);
        REQUIRE(stats.mean_f1measure() == 1.0/2);
    }

    SECTION("No relevants elements") {
        BenchmarkStats stats(5);
        stats.process(results, {});

        REQUIRE(stats.mean_precision() == 0.0);
        REQUIRE(stats.mean_recall() == 1.0);
        REQUIRE(stats.mean_f1measure() == 0.0);
    }
}

TEST_CASE("Benchmark are executed", "[benchmark]") {
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

    FeaturesIndex index(features_base);

    SECTION("Threshold: 1") {
        vector<BenchmarkStats> stats = Benchmark::Modification(index, features_modified, {1.0, 2.0});

        REQUIRE(stats[0].mean_precision() == 1.0);
        REQUIRE(stats[0].mean_recall() == 1.0);
        REQUIRE(stats[0].mean_f1measure() == 1.0);

        REQUIRE(stats[1].mean_precision() == 0.5);
        REQUIRE(stats[1].mean_recall() == 1.0);
        REQUIRE(stats[1].mean_f1measure() == 2.0/3);
    }
}