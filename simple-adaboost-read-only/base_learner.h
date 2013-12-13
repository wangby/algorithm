/// @Brief: base learner
/// @Date: 2013Äê1ÔÂ28ÈÕ 17:09:29
/// @Author: garfieldwang

#ifndef __BASE_LEARNER_H__
#define __BASE_LEARNER_H__

#include <string>

#include "ml_data.h"
#include "types.h"

class BaseLearner
{
public:
    BaseLearner(){}

    virtual ~BaseLearner(){}
public:

    virtual void Train(const Data& data, const T_VECTOR& data_dist) = 0;
    
    virtual void SaveWeights(const std::string& model_file) = 0;

    virtual void LoadWeights(const std::string& model_file) = 0;

    virtual void PredictAll(
        const Data& data,  
        const T_VECTOR& data_dist, 
        T_VECTOR& result,
        float& error
        ) = 0;

    virtual T_DTYPE PredictOne(const Data& data, int index) = 0;
private:
}; //end of class BaseLearner

class Learner1: public BaseLearner
{
public:
    Learner1();

    virtual ~Learner1(){};

    virtual void Train(const Data& data, const T_VECTOR& data_dist);

    virtual void SaveWeights(const std::string& model_file);

    virtual void LoadWeights(const std::string& model_file);

    virtual void PredictAll(
        const Data& data,  
        const T_VECTOR& data_dist, 
        T_VECTOR& result,
        float& error
        );
    
    virtual T_DTYPE PredictOne(const Data& data, int index);
    
private:

    std::string m_learn_type;
    T_DTYPE m_splitter;
    int m_best_dimension;
}; //end of class Learner1

#endif /* __BASE_LEARNER_H__ */
