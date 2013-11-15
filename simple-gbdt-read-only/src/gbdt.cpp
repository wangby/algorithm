#include <iostream>
#include <deque>
#include <algorithm>
#include <set>
#include <sstream>

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#include "gbdt.h"
#include "tbb/parallel_sort.h"

using namespace tbb;

using std::cout;
using std::endl;
using std::deque;
using std::set;
using std::string;

static bool compareNodeReduced ( nodeReduced n0, nodeReduced n1 )
{
    return n0.m_size < n1.m_size;
}

static int64_t Milliseconds()
{
    struct timeval t;    
    ::gettimeofday(&t, NULL);    
    int64_t curTime;    
    curTime = t.tv_sec;    
    curTime *= 1000;              // sec -> msec    
    curTime += t.tv_usec / 1000;  // usec -> msec    
    return curTime;
}

GBDT::GBDT()
{
    m_max_epochs = 400;
    m_global_mean = 0.0;
    m_max_tree_leafes = 20;
    m_feature_subspace_size = 40;
    m_use_opt_splitpoint = true;
    m_lrate = 0.01;
    m_train_epoch = 0;
    m_data_sample_ratio = 0.4;
}


bool GBDT::LoadConfig(const std::string& conf_file)
{
    std::ifstream fs;
    fs.open(conf_file.c_str(), std::ios_base::in);

    if (fs.fail())
    {
        std::cerr << " Warning ! Conf File isn't exist. Use default setting!" << conf_file << std::endl;
        return false;
    }
    
    std::string line;
    while (getline(fs, line))
    {
        string::size_type pos = line.find("=");
        if( pos != string::npos && pos != (line.length() - 1) )
        {
            string key = line.substr(0, pos);
            string value = line.substr(pos+1);

            std::stringstream sstream;
            if ( key == "max_epochs" )
            {
                sstream << value;
                sstream >> m_max_epochs;
            } 
            else if ( key == "max_tree_leafes" )
            {
                sstream << value;
                sstream >> m_max_tree_leafes;
            }
            else if ( key == "feature_subspace_size" )
            {
                sstream << value;
                sstream >> m_feature_subspace_size;
            }
            else if ( key == "use_opt_splitpoint" )
            {
                if ( value == "false" ) m_use_opt_splitpoint = false;
            }
            else if ( key == "learn_rate" )
            {
                sstream << value;
                sstream >> m_lrate;
            }
            else if ( key == "data_sample_ratio" )
            {
                sstream << value;
                sstream >> m_data_sample_ratio;
                if (m_data_sample_ratio > 1)
                {
                    m_data_sample_ratio = 1;
                }
                else if ( m_data_sample_ratio < 0 )
                {
                    m_data_sample_ratio = 0.01;
                }
            }
        }
    }
    return true;
}

bool GBDT::Init()
{
    m_trees = new node[m_max_epochs];
    for (unsigned int i = 0; i < m_max_epochs ; ++i )
    {
        m_trees[i].m_featureNr = -1;
        m_trees[i].m_value = 1e10;
        m_trees[i].m_toSmallerEqual = 0;
        m_trees[i].m_toLarger = 0;
        m_trees[i].m_trainSamples = 0;
        m_trees[i].m_nSamples = -1;
    }

    srand(time(0));
    cout << "configure--------" << endl;
    cout <<  "  max_epochs: " << m_max_epochs << endl;
    cout <<  "  max_tree_leafes: " << m_max_tree_leafes << endl;
    cout <<  "  feature_subspace_size: " << m_feature_subspace_size << endl;
    cout <<  "  use_opt_splitpoint: " << m_use_opt_splitpoint << endl;
    cout <<  "  learn_rate: " << m_lrate << endl;
    cout <<  "  data_sample_ratio: " << m_data_sample_ratio << endl;
    cout << endl;
    
    return true;
}

bool GBDT::Train(const Data& data)
{
    m_tree_target.resize( data.m_target.size() );
    m_feature_subspace_size = 
        m_feature_subspace_size > data.m_dimension ? data.m_dimension : m_feature_subspace_size;


    double pre_rmse = -1;
    unsigned int train_epoch =0;
    //TODO or rmse rise up
    for ( ; train_epoch < m_max_epochs; train_epoch++ )
    {
        double rmse = 0.0;
        cout << "epoch: " << train_epoch << endl;

        ModelUpdate(data, train_epoch, rmse);
        
        
        //if (pre_rmse < rmse && pre_rmse != -1)
        //if (pre_rmse - rmse < ( m_lrate * 0.001 ) && pre_rmse != -1)
        if (pre_rmse < rmse && pre_rmse != -1)
        {
            cout << "debug: rmse:" << rmse << " " << pre_rmse << " " << pre_rmse - rmse << std::endl;
            break;
        }
        pre_rmse = rmse;
    }
    m_train_epoch = train_epoch - 1;
    
    return true;
}

bool GBDT::ModelUpdate(const Data& data, unsigned int train_epoch, double& rmse)
{
    int64_t t0 = Milliseconds();

    int nSamples = data.m_num;
    unsigned int nFeatures = data.m_dimension;
    
    bool* usedFeatures = new bool[data.m_dimension];
    T_DTYPE* inputTmp = new T_DTYPE[(nSamples+1)*m_feature_subspace_size];
    T_DTYPE* inputTargetsSort = new T_DTYPE[(nSamples+1)*m_feature_subspace_size];
    int* sortIndex = new int[nSamples];
    
    //----first epoch----
    if (train_epoch == 0)
    {
        double mean = 0.0;
        if ( true )
        {
            for ( unsigned int  j=0; j< data.m_num ;j++ )
                mean += data.m_target[j];
            mean /= ( double ) data.m_num;
        }
        m_global_mean = mean;
        std::cout << "globalMean:"<< mean<<" "<<std::flush;

        //align by targets mean
        for ( unsigned int j=0 ; j<data.m_num ; j++ )
            m_tree_target[j] = data.m_target[j] - mean;
    }
    
    deque< nodeReduced > largestNodes;
    //----init feature mask----
    for ( unsigned int j=0; j< data.m_dimension; j++ )
        usedFeatures[j] = false;

    //----data should be sampled !!!!----
    int data_sample_num = int(nSamples * m_data_sample_ratio);
    if( data_sample_num < 10 ) data_sample_num = nSamples;
    
    m_trees[train_epoch].m_trainSamples = new int[data_sample_num];
    m_trees[train_epoch].m_nSamples = data_sample_num;
    int* ptr = m_trees[train_epoch].m_trainSamples;

    set<int> used_data_ids;
    int sampled_count = 0;
    while (sampled_count < data_sample_num)
    {
        int id = rand()%nSamples;
        if ( used_data_ids.find(id) == used_data_ids.end() ) //can't find the id
        {
            ptr[sampled_count] = id;
            sampled_count++;
            used_data_ids.insert(id);
        }
    }
    ///////////////////
    
    //----init first node for split----
    nodeReduced firstNode;
    firstNode.m_node = & ( m_trees[train_epoch] );
    firstNode.m_size = data_sample_num;
    largestNodes.push_back ( firstNode );
    push_heap( largestNodes.begin(), largestNodes.end(), compareNodeReduced );  //heap for select largest num node

    //----sample feature----
    int randFeatureIDs[m_feature_subspace_size];
    // this tmp array is used to fast drawing a random subset
    if ( m_feature_subspace_size < data.m_dimension ) // select a random feature subset
    {
        for ( unsigned int i=0;i<m_feature_subspace_size;i++ )
        {
            unsigned int idx = rand() % nFeatures; //
            while ( usedFeatures[idx] || (data.m_valid_id.find(idx) == data.m_valid_id.end() ) ) //TODO check valid num;
                idx = rand() % nFeatures;
            randFeatureIDs[i] = idx;
            usedFeatures[idx] = true;
        }
    }
    else  // take all features
        for ( unsigned int i=0;i<m_feature_subspace_size;i++ )
            randFeatureIDs[i] = i;

    //---- train the tree loop wise----
    // call trainSingleTree recursive for the largest node
    for ( unsigned int j=0; j<m_max_tree_leafes; j++ )
    {
        node* largestNode = largestNodes[0].m_node;

        TrainSingleTree( 
            largestNode, 
            largestNodes, 
            data, 
            usedFeatures,
            inputTmp,
            inputTargetsSort,
            sortIndex,
            randFeatureIDs
            );
        
    }
    // unmark the selected inputs
    for ( unsigned int i=0;i<nFeatures;i++ )
        usedFeatures[i] = false;
    
    //----delete the train lists per node, they are not necessary for prediction----
    cleanTree ( & ( m_trees[train_epoch] ) );

    // update the targets/residuals and calc train error
    double trainRMSE = 0.0;
    //fstream f("tmp/a0.txt",ios::out);
    for ( int j=0;j<nSamples;j++ )
    {
        T_DTYPE p = predictSingleTree ( & ( m_trees[train_epoch] ), data, j);

        //f<<p<<endl;
        m_tree_target[j] -= m_lrate * p;
        double err = m_tree_target[j];
        trainRMSE += err * err;
    }
    rmse = sqrt ( trainRMSE/ ( double ) nSamples );
    cout<<"RMSE:"<< rmse <<" " << trainRMSE << " "<<std::flush;
    cout<<"cost: " << Milliseconds() -t0<<"[ms]"<<endl;

    delete[] usedFeatures;
    delete[] inputTmp;
    delete[] inputTargetsSort;
    delete[] sortIndex;
    usedFeatures = NULL;
    inputTmp = NULL;
    inputTargetsSort = NULL;
    sortIndex = NULL;
    
    return true;
}

void GBDT::cleanTree ( node* n )
{
    if ( n->m_trainSamples )
    {
        delete[] n->m_trainSamples;
        n->m_trainSamples = 0;
    }
    n->m_nSamples = 0;

    if ( n->m_toSmallerEqual )
        cleanTree ( n->m_toSmallerEqual );
    if ( n->m_toLarger )
        cleanTree ( n->m_toLarger );
}


void GBDT::TrainSingleTree(
    node * n,
    std::deque<nodeReduced> &largestNodes,
    const Data& data,
    bool* usedFeatures,
    T_DTYPE* inputTmp, 
    T_DTYPE* inputTargetsSort,
    int* sortIndex,
    const int* randFeatureIDs
    )
{
    unsigned int nFeatures = data.m_dimension;
    
    // break criteria: tree size limit or too less training samples
    unsigned int nS = largestNodes.size();
    if ( nS >= m_max_tree_leafes || n->m_nSamples <= 1 )
        return;

    // delete the current node (is currently the largest element in the heap)
    if ( largestNodes.size() > 0 )
    {
        //largestNodes.pop_front();
        pop_heap ( largestNodes.begin(),largestNodes.end(),compareNodeReduced );
        largestNodes.pop_back();
    }

    // the number of training samples in this node
    int nNodeSamples = n->m_nSamples;

    // precalc sums and squared sums of targets
    double sumTarget = 0.0, sum2Target = 0.0;
    for ( int j=0;j<nNodeSamples;j++ )
    {
        T_DTYPE v = m_tree_target[n->m_trainSamples[j]];
        sumTarget += v;
        sum2Target += v*v;
    }

    int bestFeature = -1, bestFeaturePos = -1;
    double bestFeatureRMSE = 1e10;
    T_DTYPE bestFeatureLow = 1e10, bestFeatureHi = 1e10;
    T_DTYPE optFeatureSplitValue = 1e10;

    //TODO check m_feature_subspace_size not larger than nFeatures!!
    // search optimal split point in all tmp input features
    for ( unsigned int i=0;i<m_feature_subspace_size;i++ )
    {
        // search the optimal split value, which reduce the RMSE the most
        T_DTYPE optimalSplitValue = 0.0;
        double rmseBest = 1e10;
        T_DTYPE meanLowBest = 1e10, meanHiBest = 1e10;
        int bestPos = -1;
        double sumLow = 0.0, sum2Low = 0.0, sumHi = sumTarget, sum2Hi = sum2Target, cntLow = 0.0, cntHi = nNodeSamples;
        T_DTYPE* ptrInput = inputTmp + i * nNodeSamples;
        T_DTYPE* ptrTarget = inputTargetsSort + i * nNodeSamples;

        //  copy current feature into preInput
        int nr = randFeatureIDs[i];
        for ( int j=0;j<nNodeSamples;j++ )
            ptrInput[j] = data.m_data[ n->m_trainSamples[j] ][nr];  //line :n->m_trainSamples[j] , row:nr

        if ( m_use_opt_splitpoint == false ) // random threshold value
        {
            for ( int j=0;j<nNodeSamples;j++ )
                ptrTarget[j] = m_tree_target[n->m_trainSamples[j]];

            T_DTYPE* ptrInput = inputTmp + i * nNodeSamples;//TODO: del ????
            bestPos = rand() % nNodeSamples;
            optimalSplitValue = ptrInput[bestPos];
            sumLow = 0.0;
            sum2Low = 0.0;
            cntLow = 0.0;
            sumHi = 0.0;
            sum2Hi = 0.0;
            cntHi = 0.0;
            for ( int j=0;j<nNodeSamples;j++ )
            {
                //T_DTYPE v = ptrInput[j];
                T_DTYPE t = ptrTarget[j];
                if ( ptrInput[j] <= optimalSplitValue )
                {
                    sumLow += t;
                    sum2Low += t*t;
                    cntLow += 1.0;
                }
                else
                {
                    sumHi += t;
                    sum2Hi += t*t;
                    cntHi += 1.0;
                }
            }
            rmseBest = ( sum2Low/cntLow - ( sumLow/cntLow ) * ( sumLow/cntLow ) ) *cntLow;
            rmseBest += ( sum2Hi/cntHi - ( sumHi/cntHi ) * ( sumHi/cntHi ) ) *cntHi;
            rmseBest = sqrt ( rmseBest/ ( cntLow+cntHi ) );
            meanLowBest = sumLow/cntLow;
            meanHiBest = sumHi/cntHi;
        }
        else  // search for the optimal threshold value, goal: best RMSE reduction split
        {
            // fast sort of the input dimension
            for ( int j=0;j<nNodeSamples;j++ )
                sortIndex[j] = j;
            
            std::vector<std::pair<T_DTYPE, int> > list(nNodeSamples);
            for(int j=0;j<nNodeSamples;j++)
            {
                list[j].first = ptrInput[j];
                list[j].second = sortIndex[j];
            }
            parallel_sort(list.begin(),list.end());
            for(int j=0;j<nNodeSamples;j++)
            {
                ptrInput[j] = list[j].first;
                sortIndex[j] = list[j].second;
            }
            for ( int j=0;j<nNodeSamples;j++ )
                ptrTarget[j] = m_tree_target[n->m_trainSamples[sortIndex[j]]];
            
            int j = 0;
            while ( j < nNodeSamples-1 )
            {
                T_DTYPE t = ptrTarget[j];
                sumLow += t;
                sum2Low += t*t;
                sumHi -= t;
                sum2Hi -= t*t;
                cntLow += 1.0;
                cntHi -= 1.0;

                T_DTYPE v0 = ptrInput[j], v1 = 1e10;
                if ( j < nNodeSamples -1 )
                    v1 = ptrInput[j+1];
                if ( v0 == v1 ) // skip equal successors
                {
                    j++;
                    continue;
                }
                //rmse
                double rmse = ( sum2Low/cntLow - ( sumLow/cntLow ) * ( sumLow/cntLow ) ) *cntLow;
                rmse += ( sum2Hi/cntHi   - ( sumHi/cntHi ) * ( sumHi/cntHi ) ) *cntHi;
                rmse = sqrt ( rmse/ ( cntLow+cntHi ) );

                if ( rmse < rmseBest )
                {
                    optimalSplitValue = v0;
                    rmseBest = rmse;
                    bestPos = j+1;
                    meanLowBest = sumLow/cntLow;
                    meanHiBest = sumHi/cntHi;
                }

                j++;
            }
        }

        if ( rmseBest < bestFeatureRMSE )
        {
            bestFeature = i;
            bestFeaturePos = bestPos;
            bestFeatureRMSE = rmseBest;
            optFeatureSplitValue = optimalSplitValue;
            bestFeatureLow = meanLowBest;
            bestFeatureHi = meanHiBest;
        }
    }

    n->m_featureNr = randFeatureIDs[bestFeature];
    n->m_value = optFeatureSplitValue;

    if ( n->m_featureNr < 0 || n->m_featureNr >= (int)nFeatures )
    {
        cout<<"f="<<n->m_featureNr<<endl;
        assert ( false );
    }

    // count the samples of the low node
    int cnt = 0;
    for ( int i=0;i<nNodeSamples;i++ )
    {
        int nr = n->m_featureNr;
        if ( data.m_data[ n->m_trainSamples[i] ][nr] <= optFeatureSplitValue )
            cnt++;
    }

    int* lowList = new int[cnt];
    int* hiList = new int[nNodeSamples-cnt];
    if ( cnt == 0 )
        lowList = 0;
    if ( nNodeSamples-cnt == 0 )
        hiList = 0;

    int lowCnt = 0, hiCnt = 0;
    double lowMean = 0.0, hiMean = 0.0;
    for ( int i=0;i<nNodeSamples;i++ )
    {
        int nr = n->m_featureNr;
        if ( data.m_data[ n->m_trainSamples[i] ][nr] <= optFeatureSplitValue )
        {
            lowList[lowCnt] = n->m_trainSamples[i];
            lowMean += m_tree_target[n->m_trainSamples[i]];
            lowCnt++;
        }
        else
        {
            hiList[hiCnt] = n->m_trainSamples[i];
            hiMean += m_tree_target[n->m_trainSamples[i]];
            hiCnt++;
        }
    }
    lowMean /= lowCnt;
    hiMean /= hiCnt;
    
    if ( hiCnt+lowCnt != nNodeSamples || lowCnt != cnt )
        assert ( false );
    ///////////////////////////
    
    // break, if too less samples
    if ( lowCnt < 1 || hiCnt < 1 )
    {
        n->m_featureNr = -1;
        n->m_value = lowCnt < 1 ? hiMean : lowMean;
        n->m_toSmallerEqual = 0;
        n->m_toLarger = 0;
        if ( n->m_trainSamples )
            delete[] n->m_trainSamples;
        n->m_trainSamples = 0;
        n->m_nSamples = 0;

        nodeReduced currentNode;
        currentNode.m_node = n;
        currentNode.m_size = 0;
        largestNodes.push_back ( currentNode );
        push_heap ( largestNodes.begin(), largestNodes.end(), compareNodeReduced );

        return;
    }

    // prepare first new node
    n->m_toSmallerEqual = new node;
    n->m_toSmallerEqual->m_featureNr = -1;
    n->m_toSmallerEqual->m_value = lowMean;
    n->m_toSmallerEqual->m_toSmallerEqual = 0;
    n->m_toSmallerEqual->m_toLarger = 0;
    n->m_toSmallerEqual->m_trainSamples = lowList;
    n->m_toSmallerEqual->m_nSamples = lowCnt;

    // prepare second new node
    n->m_toLarger = new node;
    n->m_toLarger->m_featureNr = -1;
    n->m_toLarger->m_value = hiMean;
    n->m_toLarger->m_toSmallerEqual = 0;
    n->m_toLarger->m_toLarger = 0;
    n->m_toLarger->m_trainSamples = hiList;
    n->m_toLarger->m_nSamples = hiCnt;

    // add the new two nodes to the heap
    nodeReduced lowNode, hiNode;
    lowNode.m_node = n->m_toSmallerEqual;
    lowNode.m_size = lowCnt;
    hiNode.m_node = n->m_toLarger;
    hiNode.m_size = hiCnt;

    largestNodes.push_back ( lowNode );
    push_heap ( largestNodes.begin(), largestNodes.end(), compareNodeReduced );

    largestNodes.push_back ( hiNode );
    push_heap ( largestNodes.begin(), largestNodes.end(), compareNodeReduced );
    
}

T_DTYPE GBDT::predictSingleTree ( node* n, const Data& data, int data_index )
{
    int nFeatures = data.m_dimension;
    int nr = n->m_featureNr;
    if(nr < -1 || nr >= nFeatures)
    {
        cout<<"Feature nr:"<<nr<<endl;
        assert(false);
    }
    
    // here, on a leaf: predict the constant value
    if ( n->m_toSmallerEqual == 0 && n->m_toLarger == 0 )
        return n->m_value;

    //TODO : del duplicate check
    if(nr < 0 || nr >= nFeatures)
    {
        cout<<endl<<"Feature nr: "<<nr<<" (max:"<<nFeatures<<")"<<endl;
        assert(false);
    }
    T_DTYPE thresh = n->m_value;
    T_DTYPE feature = data.m_data[data_index][nr];

    if ( feature <= thresh )
        return predictSingleTree ( n->m_toSmallerEqual, data, data_index );
    return predictSingleTree ( n->m_toLarger, data, data_index );
    
}

void GBDT::PredictAllOutputs ( const Data& data, T_VECTOR& predictions)
{
    unsigned int nSamples = data.m_num;
    predictions.resize(nSamples);
    
    // predict all values 
    for ( unsigned int i=0; i< nSamples; i++ )
    {
        double sum = m_global_mean;
        // for every boosting epoch : CORRECT, but slower
        for ( unsigned int k=0; k<m_train_epoch + 1; k++ )
        {
            T_DTYPE v = predictSingleTree ( & ( m_trees[k] ), data, i );
            sum += m_lrate * v;  // this is gradient boosting
        }
        predictions[i] = sum;
    }
}

void GBDT::SaveWeights(const std::string& model_file)
{
    cout<<"Save:"<<model_file <<endl;
    std::fstream f ( model_file.c_str(), std::ios::out );

    // save learnrate
    f.write ( ( char* ) &m_lrate, sizeof ( m_lrate ) );

    // save number of epochs
    f.write ( ( char* ) &m_train_epoch, sizeof ( m_train_epoch ) );

    // save global means
    f.write ( ( char* ) &m_global_mean, sizeof ( m_global_mean ) );

    // save trees
    for ( unsigned int j=0;j<m_train_epoch+1;j++ )
        SaveTreeRecursive ( & ( m_trees[j] ), f );

    cout << "debug: train_epoch: " << m_train_epoch << endl;
    f.close();
}

void GBDT::SaveTreeRecursive ( node* n, std::fstream &f )
{
    //cout << "debug_save: " << n->m_value << " " << n->m_featureNr << endl;
    f.write ( ( char* ) n, sizeof ( node ) );
    if ( n->m_toSmallerEqual )
        SaveTreeRecursive ( n->m_toSmallerEqual, f );
    if ( n->m_toLarger )
        SaveTreeRecursive ( n->m_toLarger, f );
}

void GBDT::LoadWeights(const std::string& model_file)
{
    cout<<"Load:"<<model_file<<endl;
    std::fstream f ( model_file.c_str(), std::ios::in );
    if ( f.is_open() == false )
    {
        cout << "Load " << model_file << "failed!" << endl;
        _Exit(1);
    }

    // load learnrate
    f.read ( ( char* ) &m_lrate, sizeof ( m_lrate ) );

    // load number of epochs
    f.read ( ( char* ) &m_train_epoch, sizeof ( m_train_epoch ) );

    // load global means
    f.read ( ( char* ) &m_global_mean, sizeof ( m_global_mean )  );

    // allocate and load the trees
    m_trees = new node[m_train_epoch+1];
    for ( unsigned int j=0;j<m_train_epoch+1;j++ )
    {
        std::string prefix = "";
        LoadTreeRecursive ( & ( m_trees[j] ), f, prefix );
    }

    f.close();
}

void GBDT::LoadTreeRecursive ( node* n, std::fstream &f, std::string prefix )
{
    f.read ( ( char* ) n, sizeof ( node ) );
    
    //cout << prefix;
    //cout << "debug_load: " << n->m_value << " " << n->m_featureNr << endl;
    if ( n->m_toLarger == 0 && n->m_toSmallerEqual == 0 )
    {
        assert( n->m_featureNr == -1);
    }
    prefix += "    ";
    if ( n->m_toSmallerEqual )
    {
        n->m_toSmallerEqual = new node;
        LoadTreeRecursive ( n->m_toSmallerEqual, f , prefix);
    }
    if ( n->m_toLarger )
    {
        n->m_toLarger = new node;
        LoadTreeRecursive ( n->m_toLarger, f , prefix);
    }
}

