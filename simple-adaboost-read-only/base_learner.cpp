#include "base_learner.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

#include "assert.h"
#include "math.h"

using std::cout;
using std::endl;

Learner1::Learner1() :
    m_learn_type(""),
    m_splitter(0.0),
    m_best_dimension(-1)
{
}

void Learner1::Train(const Data& data, const T_VECTOR& data_dist)
{
    //learn by splitting the feature
    
    const size_t data_len = data.m_data.size();
    std::vector<std::pair<T_DTYPE, int> > list(data_len);

    float min_error = 1e10;
    //pre calculate value
    T_DTYPE positive_sum = 0.0;
    T_DTYPE nagetive_sum = 0.0;
    for ( size_t i = 0; i < data_len ; ++i )
    {
        if (data.m_target[i] > 0)
        {
            positive_sum += 1 * data_dist[i];
        }
        else
        {
            nagetive_sum += 1 * data_dist[i];
        }
    }
    
    for (size_t i  = 0; i < data.m_dimension; ++i )
    {
        for ( size_t j = 0; j < data_len ; ++j )
        {
            list[j].first = data.m_data[j][i];
            list[j].second = j;
        }

        sort(list.begin(), list.end());
        float error_gt = nagetive_sum; //-1 : 1
        float error_lt = positive_sum;// 1 : -1

        for ( size_t j = 0; j < data_len ; ++j )
        {
            T_DTYPE value = list[j].first;
            int index = list[j].second;
            
            T_DTYPE lable = data.m_target[index];
            //> this value  is 1
            if (lable <= 0) //match nagetive: -1 to -1
            {
                error_gt -= 1 * data_dist[index];
            }
            else //not match: 1 to -1
            {
                error_gt += 1 * data_dist[index];
            }

            //> this value is -1
            if (lable > 0) //match : 1 to 1
            {
                error_lt -= 1 * data_dist[index];
            }
            else //not match: -1 to 1
            {
                error_lt += 1 * data_dist[index];
            }

            //save the splitter
            if (error_gt < min_error)
            {
                m_splitter = value;
                m_learn_type = ">";
                min_error = error_gt;
                m_best_dimension = i;
            }
            if (error_lt < min_error)
            {
                m_splitter = value;
                m_learn_type = "<=";
                min_error = error_lt;
                m_best_dimension = i;
            }
            
        }
    }

    cout << "base learner: learn1 , " 
        << m_learn_type << ":" << m_splitter << ":" << m_best_dimension
        << " err:" << min_error 
		<< endl;
}

void Learner1::SaveWeights(const std::string& model_file)
{
    std::ofstream fout(model_file.c_str());

    if (!fout)
    {
        std::cerr << "save weights error! can't open " << model_file << endl;
        return ;
    }

    fout << m_learn_type << " " << m_splitter << " " << m_best_dimension << endl;
    fout.close();
}

void Learner1::LoadWeights(const std::string& model_file)
{
    std::ifstream fin(model_file.c_str());
    if (!fin)
    {
        std::cerr << "load weights error! can't open " << model_file << endl;
    }

    fin >> m_learn_type >> m_splitter >> m_best_dimension;
    cout << "load model: " << model_file << ", "
        << m_learn_type << ":" << m_splitter << ":" << m_best_dimension << endl;

    fin.close();
    
}

void Learner1::PredictAll(
    const Data& data,  
    const T_VECTOR& data_dist, 
    T_VECTOR& result,
    float& error
    )
{
    error = 0.0;
    size_t data_len = data.m_data.size();
    for ( size_t i = 0; i < data_len; ++i )
    {
        //todo check valid dimension
        T_DTYPE value = data.m_data[i][m_best_dimension];
        if (m_learn_type == ">" )
        {
            if ( value > m_splitter )
                result.push_back(1);
            else
                result.push_back(-1);
        }
        else if (m_learn_type == "<=" )
        {
            if ( value <= m_splitter )
                result.push_back(1);
            else
                result.push_back(-1);
        }
        else
        {
            std::cerr << "error: not a valid learn type," << m_learn_type << endl;
            assert(0);
        }
        T_DTYPE label = result[ result.size() - 1 ];
        if (label*data.m_target[i] < 0)
        {
            //predict error
            error += 1 * data_dist[i];
        }
    }

}

T_DTYPE Learner1::PredictOne(const Data& data, int index)
{
    T_DTYPE result = 0;
    T_DTYPE value = data.m_data[index][m_best_dimension];
    if (m_learn_type == ">" )
    {
        if ( value > m_splitter )
            result = 1;
        else
            result = -1;
    }
    else if (m_learn_type == "<=" )
    {
        if ( value <= m_splitter )
            result = 1;
        else
            result = -1;
    }
    else
    {
        std::cerr << "error: not a valid learn type," << m_learn_type << endl;
        assert(0);
    }
    
    return result;
}
