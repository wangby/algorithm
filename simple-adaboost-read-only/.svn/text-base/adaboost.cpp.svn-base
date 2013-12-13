#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <math.h>

#include "adaboost.h"

Adaboost::~Adaboost()
{
    ClearLearners();
}

void Adaboost::ClearLearners()
{
    typeof(m_base_learners.begin()) ite = m_base_learners.begin();
    for ( ; ite != m_base_learners.end() ; ++ite )
    {
        if (*ite != NULL)
        {
            delete *ite;
            *ite = NULL;
        }
    }
    m_base_learners.clear();
    m_weights.clear();
}

void Adaboost::SetMaxTrainEpoch(unsigned int max_train_epoch)
{
    m_max_train_epoch = max_train_epoch;
}

void Adaboost::Train(const Data& data)
{
    ClearLearners();
    //init dist;
    m_data_dist.clear();
    size_t data_len = data.m_data.size();
    m_data_dist.resize(data_len);
    for (size_t i = 0 ; i < data_len; ++i)
    {
        m_data_dist[i] = 1.0/(float)data_len;
    }
    
    for ( unsigned int i = 0 ; i < m_max_train_epoch; ++i )
    {
        //todo factory
        BaseLearner* learner = new Learner1();
        learner->Train(data, m_data_dist);

        T_VECTOR predict_results;
        float error;
        learner->PredictAll(data, m_data_dist, predict_results, error);

        if (error > 0.5)
        {
            delete learner;
            learner = NULL;
            break;
        }

        //error may be 0
        float weight = 0.5 * logf( (1 - error)/(error+0.00001) );
        m_base_learners.push_back(learner);
        m_weights.push_back(weight);

        //redistribute
        float sum = 0.0;
        for (size_t i = 0 ; i < data_len; ++i)
        {
            float sign = 1;
            if ( predict_results[i]*data.m_target[i] > 0 )
            {
                sign = -1;
            }
            m_data_dist[i] = m_data_dist[i]*expf(sign*weight);
            sum += m_data_dist[i];
        }
        //normalize to be distribution
        for (size_t i = 0 ; i < data_len; ++i)
        {
            m_data_dist[i] /= sum;
        }
        //std::cout << "debug sum " << sum << std::endl;
        //std::cout << "debug weight: " << weight << std::endl;
//         for (size_t i = 0; i < data_len; ++i )
//         {
//             std::cout << " " << m_data_dist[i] ;
//         }
//         std::cout << std::endl;
    }
}

void Adaboost::PredictAll(
    const Data& data,
    T_VECTOR& result,
    float& error
    )
{
    error = 0.0;
    size_t data_len = data.m_data.size();
    for ( unsigned int i = 0; i < data_len; ++i )
    {
        float predict_value = 0;
        for (unsigned int j = 0; j < m_base_learners.size(); ++j )
        {
            predict_value += (m_base_learners[j]->PredictOne( data, i ))*m_weights[j];
        }

        //std::cout << "debug_score: " << predict_value << std::endl;
        if (predict_value >= 0)
        {
            result.push_back(1);
        }
        else
        {
            result.push_back(-1);
        }
        
        if ( result[ result.size() - 1 ] * data.m_target[i] < 0 )
        {
            error += 1;
        }
        
    }

}

void Adaboost::SaveModels(const std::string& model_dir)
{
    std::string model_file = model_dir + "/model_";
    for ( size_t i = 0; i < m_base_learners.size(); ++i )
    {
        std::stringstream sstream;
        sstream << i;
        std::string current_model = model_file + sstream.str();

        m_base_learners[i]->SaveWeights(current_model);
    }

    std::string stat_file = model_file + "stat";
    std::ofstream fout(stat_file.c_str());

    if (!fout)
    {
        std::cerr << "save model_stat error! can't open " << stat_file << std::endl;
        return ;
    }

    fout << m_base_learners.size() << std::endl;
    for (size_t i = 0; i < m_weights.size(); ++i )
    {
        fout << m_weights[i] << std::endl;
    }
    fout.close();
    
}

void Adaboost::LoadModels(const std::string& model_dir)
{
    ClearLearners();
    
    std::string model_file = model_dir + "/model_";

    unsigned int models_size = 0;
    std::string stat_file = model_file + "stat";
    
    std::ifstream fin(stat_file.c_str());
    if (!fin)
    {
        std::cerr << "load model_stat error! can't open " << stat_file << std::endl;
    }

    fin >> models_size;
    std::cout << "num of models: " << models_size << std::endl;
    for (size_t i = 0; i < models_size; ++i )
    {
        float weight = 0.0;
        fin >> weight;
        m_weights.push_back(weight);
    }
    fin.close();
    
    
    for ( size_t i = 0; i < models_size; ++i )
    {
        std::stringstream sstream;
        sstream << i;
        std::string current_model = model_file + sstream.str();

        BaseLearner* learner = new Learner1();
        learner->LoadWeights(current_model);

        m_base_learners.push_back(learner);
    }
}

