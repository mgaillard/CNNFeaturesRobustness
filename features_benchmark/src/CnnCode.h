#ifndef CNNFEATURESBENCHMARK_CNNCODE_H
#define CNNFEATURESBENCHMARK_CNNCODE_H

#include <vector>
#include <bitset>
#include <functional>

using namespace std;

const size_t MAX_CODE_LENGTH = 16;

typedef bitset<MAX_CODE_LENGTH> CnnCode;

typedef function<float(const CnnCode&, const CnnCode&)> CnnCodeDistanceFunction;

const string CNN_DIST_HAMMING = "hamming";

CnnCodeDistanceFunction MakeCnnCodesDistanceFunction(const string& distance_type);

float CnnCodeHammingDistance(const CnnCode &code_a, const CnnCode &code_b);

#endif //CNNFEATURESBENCHMARK_CNNCODE_H
