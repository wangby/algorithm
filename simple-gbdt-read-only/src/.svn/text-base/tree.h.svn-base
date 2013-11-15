/// @Brief: tree structures define from ELF
/// @Date: 2012Äê5ÔÂ28ÈÕ 12:27:04
/// @Author: wangben

#include "types.h"


typedef struct node_
{
    int m_featureNr;                // decision on this feature
    T_DTYPE m_value;                   // the prediction value
    struct node_* m_toSmallerEqual; // pointer to node, if:  feature[m_featureNr] <=  m_value
    struct node_* m_toLarger;       // pointer to node, if:  feature[m_featureNr] > m_value
    int* m_trainSamples;            // a list of indices of the training samples in this node
    int m_nSamples;                 // the length of m_trainSamples
} node;

typedef struct nodeReduced_
{
    node* m_node;
    uint m_size;
} nodeReduced;

