#include "CnnCode.h"

CnnCodeDistanceFunction MakeCnnCodesDistanceFunction(const string& distance_type) {
    if (distance_type.compare(CNN_DIST_HAMMING) == 0) {
        return CnnCodeHammingDistance;
    }

    // By default, the distance is Hamming.
    return CnnCodeHammingDistance;
}

float CnnCodeHammingDistance(const CnnCode &code_a, const CnnCode &code_b) {
    return static_cast<float>((code_a ^ code_b).count()) / MAX_CODE_LENGTH;
}