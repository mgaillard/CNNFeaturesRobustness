#ifndef FEATURES_ROBUSTNESS_FEATURESHDF5IO_H
#define FEATURES_ROBUSTNESS_FEATURESHDF5IO_H

#include <string>
#include "Features.h"

using namespace std;

class FeaturesHdf5IO {
public:
    static Features load(const string &filename);
};


#endif //FEATURES_ROBUSTNESS_FEATURESHDF5IO_H
