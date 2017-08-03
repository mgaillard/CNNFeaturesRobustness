#ifndef CNNFEATURESBENCHMARK_FEATURESHDF5IO_H
#define CNNFEATURESBENCHMARK_FEATURESHDF5IO_H

#include <string>
#include <vector>
#include "CnnFeatures.h"
#include "CnnCode.h"

using namespace std;

class FeaturesHdf5IO {
public:
    static vector<CnnFeatures> loadFeatures(const string &filename);

    static vector<CnnCode> loadBinaryCodes(const string &filename);
};


#endif //CNNFEATURESBENCHMARK_FEATURESHDF5IO_H
