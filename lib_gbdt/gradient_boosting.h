#include "stdio.h"
#include "string.h"
#include "memory.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include <getopt.h>

#include <string>

using namespace std;

#define GBDT_TERMINAL -1
#define GBDT_TOSPLIT  -2
#define GBDT_INTERIOR -3

#define uint32 unsigned int
#define swap_int(a, b) ((a ^= b), (b ^= a), (a ^= b))

#define SAMPLE_TYPE 1
//#define SAMPLE_RATIO 1.0

#define BUFFER_LENGTH 10240

#define NO_VALUE 0x7FFFFFFF

#define LOG_ERROR_(message) fprintf(stderr, "%s:%d:%s(): %s\n", __FILE__, __LINE__, __FUNCTION__, message); 
#define LOG_WARNING_(message) fprintf(stderr, "%s:%d:%s(): %s\n", __FILE__, __LINE__, __FUNCTION__, message); 
#define LOG_NOTICE_(message) fprintf(stderr , "%s\n", message);


typedef struct {
	int* nodestatus;	//!<  
	int* depth;			// 
	int* splitid;		//!< 
	double* splitvalue; //!< 
	int* ndstart;		//!< 节点对应于 Index 的开始位置
	int* ndcount;		//!< 节点内元素的个数
	double* ndavg;		//!< 节点内元素的均值 
	int* lson;			//!< 
	int* rson;			//!< 
	int nodesize;		//!< 树的节点个数
}gbdt_tree_t;


typedef struct
{
	int tree_num; //!< 森林树的个数
	int fea_num; //!< Feature的数量
	int data_num; //!< 训练数据的数据量
	int sample_num; //!< 训练数据的采样量
	int rand_fea_num; //!< Feature的采样数量

	double shrink; //!< 学习率
	int gbdt_min_node_size; //!< 树停止的条件，节点覆盖的最少的数据量
	int gbdt_max_depth; //!< 树停止的条件，树的深度

	char train_filename[BUFFER_LENGTH]; //!< 训练样本的文件名
	char model_filename[BUFFER_LENGTH]; //!< 模型文件名
} gbdt_info_t; //!< 模型配置参数的结构体


typedef struct
{
	gbdt_tree_t** reg_forest; //!< 回归森林
	gbdt_info_t info; //!< GBDT的配置参数

	double* feature_average; //!< Feature在训练数据的平均值
}gbdt_model_t; //!< GBDT模型的结构体


typedef struct  
{
	int* fea_pool; //!< 随机 feature 候选池
	double* fvalue_list; //!< 以feature i 为拉链的特征值 x_i
	double* fv; //!< 特征值排序用的buffer版本
	double* y_list; //!< 回归的y值集合
	int* order_i; //!< 排序的标号
} bufset; //!< 训练数据池


typedef struct 
{
	int index_b; //!< 节点覆盖数据的开始
	int index_e; //!< 节点覆盖数据的结束
	int nodenum; //!< 节点覆盖的数据量
	double nodesum; //!< 节点覆盖的数据y值的和，回归用
	double critparent; //!< 分裂的评价值

} nodeinfo; //!< 节点的信息


typedef struct  
{
	int bestid; //!< 分裂使用的Feature ID
	double bestsplit; //!< 分裂边界的x值
	int pivot; //!< 分裂边界的数据标号 
} splitinfo; //!< 分裂的信息

 /*
  * @brief 在训练数据中遍历随机抽取的Feature寻找分割的最佳位置
  *
  * @param <IN> gbdt_inf : 模型的配置信息结构体
  * @param <IN> data_set : 训练数据池
  * @param <IN> x_fea_value : 训练数据中的feature值
  * @param <IN> y_score : 训练数据中的所有目标值
  * @param <IN> ninf : 根节点信息
  * @param <IN> index : 排序的序号
  * @param <IN> spinf : 分裂的节点信息
  * @return=-1 : 分裂失败
  * @return=1 : 分裂的特殊情况，无法选出可以分割的Feature
  * @return=0 : 分裂成功
  * 
  */
 int gbdt_tree_node_split(gbdt_info_t gbdt_inf, bufset* data_set, double *x_fea_value, double *y_score,
		      nodeinfo ninf, int* index, splitinfo* spinf);

 int gbdt_single_tree_estimation(double *x_fea_value, double *y_gradient, 
 gbdt_info_t gbdt_inf, bufset* data_set, 
 int* index, gbdt_tree_t* gbdt_single_tree, int nrnodes );


 int gbdt_regression_predict(gbdt_model_t* gbdt_model, double *x_test, double& ypredict);
  
 int gbdt_tree_predict(double *x_test, gbdt_tree_t *gbdt_single_tree, double& ypred, double shrink);

 gbdt_model_t* gbdt_regression_train(double *x_fea_value, double *y_result_score, gbdt_info_t infbox);
 int read_conf_file(gbdt_info_t& infbox, int argc, char* argv[]);
 int splitline(string line, string items[], int items_num, const char separator);
 int gbdt_save_model(gbdt_model_t* gbdt_model, char* model_filename);
 gbdt_model_t* gbdt_load_model(char* model_file);


 int free_model(gbdt_model_t*& gbdt_model);
  
 void R_qsort_I(double *v, int *I, int i, int j);
















