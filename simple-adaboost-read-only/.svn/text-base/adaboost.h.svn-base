/// @Brief: easy adaboost {+1, -1}
/// @Date: 2013Äê1ÔÂ28ÈÕ 17:00:24
/// @Author: garfieldwang

#ifndef __ADABOOST_H__
#define __ADABOOST_H__
#include <vector>

#include "ml_data.h"
#include "base_learner.h"

class Adaboost
{
public:
    Adaboost() :
        m_max_train_epoch(100)
    {
    }

    ~Adaboost();

public:

    void SetMaxTrainEpoch(unsigned int max_train_epoch);
    
    void Train(const Data& data);

    void PredictAll(
        const Data& data,
        T_VECTOR& result,
        float& error
        );

    void SaveModels(const std::string& model_dir);
    void LoadModels(const std::string& model_dir);
    
private:
    
    void ClearLearners();

    unsigned int m_max_train_epoch;
    std::vector<BaseLearner *> m_base_learners;
    std::vector<float> m_weights;
    T_VECTOR m_data_dist;
    
}; //end of class Adaboost

#endif /* __ADABOOST_H__ */
