#include <iostream>
#include <string>

#include "adaboost.h"

using namespace std;


void Usage()
{
    cout << "adaboost by wangben 20130128" << endl;
    cout << "Usage: " << endl;
    cout << "  -i  input file " << endl;
    cout << "  -t or -p  act type(t for train,p for predict): "  << endl;
    cout << "  -m  model files dir, default './models' "  << endl;
    cout << "  -e  max train epoch " << endl;
    cout << endl;
}

int main(int argc, char* argv[])
{
    string data_file = "";
    string act_type = "";
    string models_dir = "./models";
    unsigned int max_epoch = 100;
    //----parse command line
    int opt_c;
    while ( (opt_c = getopt( argc, argv, "i:m:e:tp")) != EOF )
    {
        switch (opt_c)
        {
        case 'i':
            data_file = optarg;
            break;
        case 'm':
            models_dir = optarg;
            break;
        case 't':
            act_type = "t";
            break;
        case 'p':
            act_type = "p";
            break;
        case 'e':
            max_epoch = atoi(optarg);
        default:
            break;
        }
        
    }

    //check options
    if ( act_type.length() == 0 
        || data_file.length() == 0 )
    {
        std::cerr << "miss parameter!!" << endl;
        Usage();
        return 1;
    }
    
    Data data;
    DataReader reader;
    reader.ReadDataFromCVS(data_file, data);
    Adaboost learner;
    
    if (act_type == "t")
    {
        learner.SetMaxTrainEpoch(max_epoch);
        learner.Train(data);

        float error;
        T_VECTOR result;
        learner.PredictAll(data, result, error);
        /*
        cout << "Predict Result:" << endl;
        for (unsigned int i = 0; i < result.size(); ++i )
        {
            cout << result[i] << endl;
        }
        */
        cout << "train err: " << error << " error rate:" << error/float(data.m_num) << endl;
        cout << "saving models to " << models_dir << endl;
        learner.SaveModels(models_dir);
    }
    else if (act_type == "p")
    {
        learner.LoadModels(models_dir);
        float error;
        T_VECTOR result;
        learner.PredictAll(data, result, error);
        cout << "test err: " << error << " error rate:" << error/float(data.m_num) << endl;
    }
    return 0;
}


