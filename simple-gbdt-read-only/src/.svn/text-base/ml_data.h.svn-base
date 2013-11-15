/// @Brief: read dataset from file
/// @Date: 2012Äê5ÔÂ28ÈÕ 11:18:27
/// @Author: wangben

#ifndef __ML_DATA_H__
#define __ML_DATA_H__

#include <vector>
#include <string>
#include <set>

#include "types.h"

class Data
{
public:
    Data(){}

    ~Data(){}
public:
    T_MATRIX m_data;
    T_VECTOR m_target;

    std::set<int>    m_valid_id;

    unsigned int m_dimension;
    unsigned int m_num;
private:
}; //end of class Data

class DataReader
{
public:
    DataReader(){}

    ~DataReader(){}

    bool ReadDataFromL2R(const std::string& input_file, Data& data, unsigned int dimentions);

    bool ReadDataFromCVS(const std::string& input_file, Data& data);
    
private:
}; //end of class DataReader


#endif /* __ML_DATA_H__ */
