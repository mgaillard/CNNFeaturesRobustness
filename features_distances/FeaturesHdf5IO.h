#ifndef CNNFEATURESDISTANCES_FEATURESHDF5IO_H
#define CNNFEATURESDISTANCES_FEATURESHDF5IO_H

#include <string>
#include <vector>
#include "CnnFeatures.h"

using namespace std;

class FeaturesHdf5IO {
public:
    static vector<CnnFeatures> load(const string &filename);
};


#endif //CNNFEATURESDISTANCES_FEATURESHDF5IO_H
