/// @Brief: A simple implement of gbrt, parallelization with tbb lib, by wangben 2012
/// @Date: 2012Äê5ÔÂ28ÈÕ 11:10:04
/// @Author: wangben
/// @address: yihucha166@yahoo.com.cn

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#include <unistd.h>

#include "types.h"
#include "ml_data.h"
#include "gbdt.h"

using namespace std;

void Usage()
{
    cout << "A simple implement of gbrt, parallelization with tbb lib, by wangben 2012" << endl;
    cout << "Usage: " << endl;
    cout << "  -i  input file " << endl;
    cout << "  -f  [cvs, l2r]  input format , default 'l2r'" << endl;
    cout << "  -c  gbrt model config file , default './gbrt.conf' "  << endl;
    cout << "  -t or -p  act type(t for train,p for predict): "  << endl;
    cout << "  -m  model file , default './gbrt.model' "  << endl;
    cout << "  -d  max dimention(only for L2R format) " << endl;
    cout << endl;
    cout << "Example: " << endl;
    cout << "train example: ./gbrt -t -i train_l2r.dat -d 415" << endl;
    cout << "predict example: ./gbrt -p -i test_l2r.dat -d 415 -m gbrt.model" << endl;
    cout << endl;
    cout << "Input Format:" << endl;
    cout << "  l2r: \"-1\tqid:1234\t1:1\t3:1\t8:1\"" << endl;
    cout << "  cvs: \"1,3,-1\" last column is label" << endl;
    cout << endl;
}

int main(int argc, char ** argv)
{
    std::string input_file = "";
    std::string input_type = "l2r";
    std::string config_file = "./gbrt.conf";
    std::string act_type = "";
    std::string model_file = "./gbrt.model";
    int dimention = 415;
    
    //----parse command line
    int opt_c;
    while ( (opt_c = getopt( argc, argv, "d:f:i:c:m:tp")) != EOF )
    {
        switch (opt_c)
        {
        case 'i':
            input_file = optarg;
            break;
        case 'f':
            input_type = optarg;
            break;
        case 'c':
            config_file = optarg;
            break;
        case 'm':
            model_file = optarg;
            break;
        case 't':
            act_type = "t";
            break;
        case 'p':
            act_type = "p";
            break;
        case 'd':
            dimention = atoi(optarg);
        default:
            break;
        }
        
    }

    //check options
    if ( act_type.length() == 0 
        || input_file.length() == 0 )
    {
        std::cerr << "miss parameter!!" << endl;
        Usage();
        return 1;
    }
    else
    {
        cout << "parameters--------" << endl;
        cout << "  input file: " << input_file << endl;
        cout << "  input format (cvs, l2r): " << input_type<< endl;
        cout << "  config file: " << config_file << endl;
        cout << "  act type(t for train,p for predict): " << act_type << endl;
        cout << "  model file: " << model_file << endl;
        cout << "  max dimention(for L2R format): " << dimention << endl;
        cout << endl;
    }

    Data data;
    DataReader dr;

    if ( input_type == "cvs")
    {
         if ( false == dr.ReadDataFromCVS(input_file, data))
         {
             std::cerr << "error: read CVS file failed! " << input_file << std::endl;
             return 1;
         }
    }
    else
    {
        if ( false == dr.ReadDataFromL2R(input_file, data, dimention))
        {
         std::cerr << "error: read L2R file failed! " << input_file << std::endl;
         return 1;
        }
    }

    GBDT gbdt;

    if (!gbdt.LoadConfig(config_file))
        return 1;
    
    if (act_type == "t")
    {
        gbdt.Init();
        gbdt.Train(data);
        gbdt.SaveWeights(model_file);
    }
    else if( act_type == "p" )
    {
        T_VECTOR predictions;
        gbdt.LoadWeights(model_file);
        gbdt.PredictAllOutputs(data, predictions);
        
        //----output prediction----
        std::ifstream fs;
        fs.open(input_file.c_str(), std::ios_base::in);

        std::string prediction_file = input_file + ".prediction";
        std::fstream fs_out;
        fs_out.open(prediction_file.c_str(), std::ios_base::out);
        
        std::string strLine;
        unsigned int line_num = 0;
        while (getline(fs, strLine))
        {
             if (strLine.length() < 2)
             {
                 continue;
             }
             fs_out<< predictions[line_num] << std::endl;
             //for debug
             //cout << strLine << "\t" << predictions[line_num] << std::endl;
             line_num++;
        }

        fs.close();
        
    }

    
    return 0;
}

