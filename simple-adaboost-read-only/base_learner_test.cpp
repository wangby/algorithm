#include <iostream>
#include <string>

#include "base_learner.h"
#include "ml_data.h"
#include "assert.h"

using namespace std;

int main(int argc, char ** argv)
{
    string data_file = argv[1];

    Data data;
    DataReader reader;
    reader.ReadDataFromCVS(data_file, data);

    size_t data_len = data.m_data.size();
    T_VECTOR data_dist;
    data_dist.resize(data_len);
    for (size_t i = 0 ; i < data_len; ++i)
    {
        data_dist[i] = 1.0/(float)data_len;
    }
    
    Learner1 test_learn;
    test_learn.Train(data, data_dist);
    string model_file = "./test_model.dat";
    test_learn.SaveWeights(model_file);
    test_learn.LoadWeights(model_file);

    T_VECTOR result;
    float error;
    test_learn.PredictAll(data, data_dist, result, error);
    assert(result.size() == data.m_data.size());

    cout << "PredictAll err: " << error << endl;
    
    return 0;
}


