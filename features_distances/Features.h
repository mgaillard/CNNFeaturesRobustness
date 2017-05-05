#ifndef FEATURES_ROBUSTNESS_FEATURES_H
#define FEATURES_ROBUSTNESS_FEATURES_H

#include <vector>

using namespace std;

class Features {
public:
    Features(const vector<vector<float> > &features) :
            features_(features) {}

    unsigned long nb_images() const { return features_.size(); };

    const vector<float> &image_features(unsigned int image) const { return features_[image]; };

private:
    vector<vector<float> > features_;
};


#endif //FEATURES_ROBUSTNESS_FEATURES_H
