/// @Brief: gbdt class
/// @Date: 2012Äê5ÔÂ28ÈÕ 12:27:04
/// @Author: wangben

#ifndef __GBDT_H__
#define __GBDT_H__

#include <deque>
#include <string>
#include <fstream>

#include "tree.h"
#include "ml_data.h"

class GBDT
{
public:
    GBDT();

    ~GBDT(){}

    bool Init();

    bool Train(const Data& data);

    void PredictAllOutputs ( const Data& data, T_VECTOR& predictions);

    void SaveWeights(const std::string& model_file);

    void LoadWeights(const std::string& model_file);

    bool LoadConfig(const std::string& conf_file);
    
private:
    bool ModelUpdate(const Data& data, unsigned int train_epoch, double& rmse);
    
    void TrainSingleTree(
        node * n,
        std::deque<nodeReduced> &largestNodes,
        const Data& data,
        bool* usedFeatures,
        T_DTYPE* inputTmp, 
        T_DTYPE* inputTargetsSort,
        int* sortIndex,
        const int* randFeatureIDs
        );
    
    T_DTYPE predictSingleTree ( 
        node* n, 
        const Data& data, 
        int data_index 
        );
    
    void cleanTree ( node* n );
    
    void SaveTreeRecursive ( node* n, std::fstream &f );
    void LoadTreeRecursive ( node* n, std::fstream &f , std::string prefix);
private:
    node * m_trees;
    unsigned int m_max_epochs;
    unsigned int m_max_tree_leafes;
    unsigned int m_feature_subspace_size;
    bool m_use_opt_splitpoint;
    double m_lrate;
    unsigned int m_train_epoch;
    float m_data_sample_ratio;
    
    T_VECTOR m_tree_target;

    T_DTYPE m_global_mean;
    
}; //end of class GBDT


#endif /* __GBDT_H__ */
