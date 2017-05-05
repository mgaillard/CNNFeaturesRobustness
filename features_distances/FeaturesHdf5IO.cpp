#include <H5Cpp.h>
#include "FeaturesHdf5IO.h"

using namespace std;
using namespace H5;

Features FeaturesHdf5IO::load(const string &filename) {
    H5File hdf5_file(filename, H5F_ACC_RDONLY);

    DataSet dataset = hdf5_file.openDataSet("features");
    DataSpace dataspace = dataset.getSpace();
    // Get the dimension size of each dimension and allocate a buffer.
    hsize_t dims_out[2];
    int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
    unsigned long nb_images = dims_out[0];
    unsigned long features_size = dims_out[1];
    // Read the data
    float *buffer = new float[nb_images * features_size];
    dataset.read(buffer, PredType::NATIVE_FLOAT);

    dataspace.close();
    dataset.close();
    hdf5_file.close();

    vector<vector<float> > features_data(nb_images, vector<float>(features_size));

    for (unsigned long i = 0; i < nb_images; i++) {
        for (unsigned long j = 0; j < features_size; j++) {
            features_data[i][j] = buffer[i * features_size + j];
        }
    }

    delete[] buffer;

    return Features(features_data);
}