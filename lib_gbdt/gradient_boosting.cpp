#include "gradient_boosting.h"

#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

int max_feature_label(string line)
{
	int start = 0;
	int fid;
	int max_fid = 0;
	int len = line.length();
 
	for(int i = 0; i < len; i++) {
		if(line[i] == ' ') {
			start = i+1;
		}
		else if(line[i] == ':') {
			if(sscanf(line.substr(start, i - start).c_str(), "%d", &fid) == 1) {
				if(max_fid < fid) {
					max_fid = fid;
				}
			}	
		}
	}
 
	return max_fid;
}
 

int splitline(string line, string items[], int items_num, const char separator)
{
	if(items == NULL || items_num <= 0) {
		return -1;
 	}

	int n = line.length();
 	int j = 0;
 	int start = 0;

 	for(int i = 0; i < n; i++) {
		if(line[i] == separator) {
			if(j < items_num && start < n) {
				items[j++] = line.substr(start, i-start);
				start = ++i;
			}
		}
 	}

 	if(j < items_num && start < n) {
		items[j++] = line.substr(start, n-start);
 	}

 	return j;
}

int gbdt_tree_node_split( 
		gbdt_info_t gbdt_inf, 
		bufset* data_set, 
		double *x_fea_value, 
		double *y_score,
	    nodeinfo ninf, 
		int* index, 
		splitinfo* spinf)
{
 	spinf->bestsplit = -1;  // 当前节点选定的特征分裂值
 	spinf->bestid = -1;     // 当前节点选定的分裂特征

 	for (int i=0; i < gbdt_inf.fea_num; ++i) { // 初始化特征池
		data_set->fea_pool[i] = i;
 	}
 	int last = gbdt_inf.fea_num - 1; //从0开始，连续的

 	double critmax = 0.0; // 用于选定最优的特征
 	for (int i = 0; i < gbdt_inf.rand_fea_num; ++i) // 随机选择rand_fea_num个特征进行分裂，并确定最优的一个分裂特征
 	{
		int select = rand() % (last+1);  // 随机选择一个特征
 		int fid = data_set->fea_pool[select]; // fid = 选出的id 0-max_id
 		data_set->fea_pool[select] = data_set->fea_pool[last]; // 跟最后一个交换，不再选 
 		data_set->fea_pool[last] = fid;
 		last--;
        // 取这个featrue所有的value值，排序
		for (int j = ninf.index_b; j <= ninf.index_e; j++){
			data_set->fvalue_list[j] = x_fea_value[index[j]* gbdt_inf.fea_num + fid]; // index[j]样本编号，读取所有样本该维特征的值
 		 	data_set->fv[j] = data_set->fvalue_list[j];  // 复制一份，用于排序
 		 	data_set->y_list[j] = y_score[index[j]];  // 提取 y 值
 		}
 		
 		for (int j = 0; j < gbdt_inf.sample_num; ++j) {
			data_set->order_i[j] = j;  // 当前树所用到的样本下标
 		}
 		
 		R_qsort_I(data_set->fv, data_set->order_i, ninf.index_b+1 , ninf.index_e+1);
 	
		if (data_set->fv[ninf.index_b] >= data_set->fv[ninf.index_e]) { //起始特征值大于等于结束特征值
			continue; // x的fid特征取值都一样，那这个特征没有区分性。。。
 	 	}
 	 	
        // 选取当前特征的最优分裂值
		double left_sum = 0.0;
 	 	int left_num = 0;
 	 	double right_sum = ninf.nodesum;
 	 	int right_num = ninf.nodenum;
 	 	double d = 0.0;
 	 	double crit = 0.0;
 	 	double tmpsplit = 0;
 	 	double critvar = 0;
 	
		for (int j=ninf.index_b; j< ninf.index_e; j++)
 	 	{
 	 	// d = y_result_score[data_set->order_i[j]];
			d = data_set->y_list[data_set->order_i[j]];  // 样本下标也对应上述特征升序排列发生了变化, e.g. j=0, order_i[j]=48
 	 		left_sum  += d;	left_num++;
 	 		right_sum -= d;	right_num--;
 	 	
			if (data_set->fv[j] < data_set->fv[j+1]) {  // 依次比较相邻的两个样本（已排序），找到最优的分裂值
				crit = (left_sum * left_sum / left_num) + (right_sum * right_sum / right_num) - ninf.critparent;
 	 			if (crit > critvar) {
					tmpsplit = (data_set->fv[j] + data_set->fv[j+1]) / 2.0;  // 实际分割用的feature value
					critvar = crit;
				}
 	 		}
		}
 	
		if (critvar > critmax) {  // 选择从开始到当前循环，最好的分裂特征及分裂值, 如果这个feature最终的critvar > cirtmax, 保存信息
			spinf->bestsplit = tmpsplit; // split feature vale 
 	 		spinf->bestid = fid;         // split feature id
 	 		critmax = critvar;           // split crit vaule
 	 	}
	}
 	
 	if( spinf->bestid >= 0)  // 根据bestid分裂node，调整index
 	{
		int nl = ninf.index_b;
		for (int j= ninf.index_b; j<= ninf.index_e; j++)  // 调整左子节点的样本下标
		{	
			if (x_fea_value[index[j]* gbdt_inf.fea_num + spinf->bestid] <= spinf->bestsplit)
			{
				data_set->order_i[nl] = index[j]; //update data->set
				nl++;
			}
		}
		int nr = nl; 
		for (int j= ninf.index_b; j<= ninf.index_e; j++) // 调整右子节点的样本
		{
			if (x_fea_value[index[j]* gbdt_inf.fea_num + spinf->bestid] > spinf->bestsplit)
			{
				data_set->order_i[nr] = index[j];
				nr++;
			}
		}
		// 将调整后的样本赋值到index，进行下一次训练
		for (int j= ninf.index_b; j<= ninf.index_e; j++)
		{
			index[j] = data_set->order_i[j];
		}
 	
		spinf->pivot = nl;  // 存储当前分裂节点，e.g. 785
 	
		return 0;
	}
 	else
 	{
		return 1;
 	}
}



int gbdt_single_tree_estimation(
		double *x_fea_value, 
		double *y_gradient, 
		gbdt_info_t gbdt_inf, 
		bufset* data_set, 
		int* index, 
		gbdt_tree_t* gbdt_single_tree, 
		int nrnodes )
{
	splitinfo* spinf = (splitinfo*) malloc(sizeof(splitinfo));
 	spinf->bestid = -1;  // 最优分隔特征id

	//for (int i = 0; i < gbdt_inf.sample_num; ++i) 	// 初始赋值，训练本棵树所用的的样本下表 ，0,1,2,...,1499
	//	index[i] = i;  //不屏蔽，每次都选一样的样本。。。

	int ncur = 0;  
	// 根节点初始化
 	gbdt_single_tree->nodestatus[0] = GBDT_TOSPLIT;
 	gbdt_single_tree->ndstart[0]	= 0;
 	gbdt_single_tree->ndcount[0]	= gbdt_inf.sample_num;
 	gbdt_single_tree->depth[0]		= 0;

	/* compute mean and sum of squares for Y */
 	double avg = 0.0;
 	for (int i = 0; i < gbdt_inf.sample_num; ++i)  // 计算训练样本目标值的均值
		avg = (i * avg + y_gradient[index[i]]) / (i + 1); 
 	

	gbdt_single_tree->ndavg[0] = avg; // 节点内元素的均值

	// 如果当前(根)节点的样本数少于节点分裂要求的样本数，则分解终止
	if (gbdt_single_tree->ndcount[0] <= gbdt_inf.gbdt_min_node_size) 
 	{
		gbdt_single_tree->nodestatus[0] = GBDT_TERMINAL;
 		gbdt_single_tree->lson[0]		= 0; // debug temp
 		gbdt_single_tree->rson[0]		= 0;
 		gbdt_single_tree->splitid[0]	= 0;
 		gbdt_single_tree->splitvalue[0] = 0.0;

 		gbdt_single_tree->nodesize		= 1;
 		free(spinf);
 		return 0;
 	}

	/* start main loop */ 
	//nrnodes是训练当前树的样本数的2倍+1，e.g. 3001
 	for (int k = 0; k < nrnodes - 2; ++k) {
		if (k > ncur || ncur >= nrnodes - 2) {
			break;
		}

		/* skip if the node is not to be split */   
		
 		if (gbdt_single_tree->nodestatus[k] != GBDT_TOSPLIT) {
			continue;
 		}

		/* initialize for next call to findbestsplit */
 		nodeinfo ninf;
 		ninf.index_b = gbdt_single_tree->ndstart[k]; // begin,起始样本下标
 		ninf.index_e = gbdt_single_tree->ndstart[k] + gbdt_single_tree->ndcount[k] - 1; // end ,结束
 		ninf.nodenum = gbdt_single_tree->ndcount[k];  // 节点样本数
 		ninf.nodesum = gbdt_single_tree->ndcount[k] * gbdt_single_tree->ndavg[k];
 		ninf.critparent = (ninf.nodesum * ninf.nodesum) / ninf.nodenum; // 节点均方

 		int jstat;
        // 决策树分裂，随机选择rand_fea_num个特征尝试对某一节点分裂
        // 貌似是以分裂后的均方误差最小化作为评判依据
		jstat = gbdt_tree_node_split(gbdt_inf, data_set, x_fea_value, y_gradient, ninf, index, spinf);

 		if (jstat == 1) // 基本不会出现
 		{
			/* Node is terminal: Mark it as such and move on to the next. */
 			gbdt_single_tree->nodestatus[k] = GBDT_TERMINAL;
 			continue;
 		}
 		if(jstat == -1)
 		{
			free(spinf);
			return -1;
 		}

 		gbdt_single_tree->splitid[k] = spinf->bestid; // 当前节点的分裂特征，e.g. k=0(根节点)   
 		gbdt_single_tree->splitvalue[k] = spinf->bestsplit;  // 特征分裂值，e.g. 344112
 		gbdt_single_tree->nodestatus[k] = GBDT_INTERIOR;

		/* leftnode no.= ncur+1, rightnode no. = ncur+2. */
		// 左子节点开始、结束样本下标，右子节点开始、结束样本下标，深度+1 
 		gbdt_single_tree->ndstart[ncur + 1] = ninf.index_b;
 		gbdt_single_tree->ndstart[ncur + 2] = spinf->pivot;
 		gbdt_single_tree->ndcount[ncur + 1] = spinf->pivot - ninf.index_b; // 不包括pivot
 		gbdt_single_tree->ndcount[ncur + 2] = ninf.index_e - spinf->pivot + 1;

 		gbdt_single_tree->depth[ncur + 1] = gbdt_single_tree->depth[k] + 1;
 		gbdt_single_tree->depth[ncur + 2] = gbdt_single_tree->depth[k] + 1;

 		/* compute mean and sum of squares for the left son node */
 		 // 计算左右子节点的sum和square
 		double avg = 0.0;
 		double d   = 0.0;
 		int m = 0;
 		for (int j = ninf.index_b; j < spinf->pivot; ++j) // mean
 		{
			d = y_gradient[index[j]];
			m = j - ninf.index_b;
			avg = (m * avg + d) / (m+1);
 		}

		double var = 0.0;
 		for (int j = ninf.index_b; j < spinf->pivot; ++j)
 		{
			var += (y_gradient[index[j]] - avg) * (y_gradient[index[j]] - avg);
 		}
 		var /= (spinf->pivot - ninf.index_b);

		gbdt_single_tree->ndavg[ncur+1] = avg;
 		gbdt_single_tree->nodestatus[ncur+1] = GBDT_TOSPLIT;
 		if (gbdt_single_tree->ndcount[ncur + 1] <= gbdt_inf.gbdt_min_node_size) 
 		{
			gbdt_single_tree->nodestatus[ncur + 1] = GBDT_TERMINAL;
 			gbdt_single_tree->lson[ncur + 1]       = 0; // debug temp
 			gbdt_single_tree->rson[ncur + 1]       = 0;
 			gbdt_single_tree->splitid[ncur + 1]    = 0;
 			gbdt_single_tree->splitvalue[ncur + 1] = 0.0;
 		}

		if (gbdt_single_tree->depth[ncur + 1] >= gbdt_inf.gbdt_max_depth) 
 		{
			gbdt_single_tree->nodestatus[ncur + 1] = GBDT_TERMINAL;
 			gbdt_single_tree->lson[ncur + 1]       = 0; // debug temp
 			gbdt_single_tree->rson[ncur + 1]       = 0;
 			gbdt_single_tree->splitid[ncur + 1]    = 0;
 			gbdt_single_tree->splitvalue[ncur + 1] = 0.0;
 		}

 		/* compute mean and sum of squares for the right daughter node */
 		avg = 0.0;
 		d   = 0.0;
 		m   = 0;
 		for (int j = spinf->pivot; j <= ninf.index_e; ++j) { 
			d   = y_gradient[index[j]];
 			m   = j - spinf->pivot;
 			avg = (m * avg + d) / (m + 1);
 		}
 		var = 0.0;
 		for (int j = spinf->pivot; j <= ninf.index_e; ++j) 
 		{
			var += (y_gradient[index[j]] - avg) * (y_gradient[index[j]] - avg);
 		}
 		var /= (ninf.index_e - spinf->pivot +1);

 		gbdt_single_tree->ndavg[ncur+2] = avg;
 		gbdt_single_tree->nodestatus[ncur+2] = GBDT_TOSPLIT;

 		if (gbdt_single_tree->ndcount[ncur + 2] <= gbdt_inf.gbdt_min_node_size) 
 		{
			gbdt_single_tree->nodestatus[ncur + 2] = GBDT_TERMINAL;
 			gbdt_single_tree->lson[ncur + 2]       = 0; // debug temp
 			gbdt_single_tree->rson[ncur + 2]       = 0;
 			gbdt_single_tree->splitid[ncur + 2]    = 0;
 			gbdt_single_tree->splitvalue[ncur +2]  = 0.0;
 		}

 		if (gbdt_single_tree->depth[ncur + 2] >= gbdt_inf.gbdt_max_depth) 
 		{
			gbdt_single_tree->nodestatus[ncur + 2] = GBDT_TERMINAL;
 			gbdt_single_tree->lson[ncur + 2]       = 0; // debug temp
 			gbdt_single_tree->rson[ncur + 2]       = 0;
 			gbdt_single_tree->splitid[ncur + 2]    = 0;
 			gbdt_single_tree->splitvalue[ncur +2]  = 0.0;
 		}

		// 可以考虑计算这个节点的方差,如果方差到了一定的阈值，停止分裂
		// if (var <= xxx)
 		// {
 		//    gbdt_single_tree->nodestatus[ncur + 2] = GBDT_TERMINAL;
 		// }
		//

 		gbdt_single_tree->lson[k] = ncur +1;
 		gbdt_single_tree->rson[k] = ncur +2;

 		ncur += 2;
	}

	gbdt_single_tree->nodesize = ncur+1;

	free(spinf);

	return 0;
}
  
int gbdt_regression_predict(gbdt_model_t* gbdt_model, double *x_test, double& ypredict)
{
	if(x_test == NULL || gbdt_model == NULL || gbdt_model->reg_forest == NULL)
 	{
		LOG_ERROR_("Parameter error.");
		return -1;
 	}

 	ypredict = 0.0;
 	for (int i=0; i<gbdt_model->info.tree_num; i++)
 	{
		if(gbdt_model->reg_forest[i] != NULL)
 		{
			if(gbdt_tree_predict(x_test, gbdt_model->reg_forest[i], ypredict, gbdt_model->info.shrink) != 0)
			{
			return -1;
			}
 		}
 		else
			return -1;
 	}
 	return 0;
}
  
int gbdt_tree_predict(double *x_test, gbdt_tree_t *gbdt_single_tree, double& ypred, double shrink)
{
	if(x_test == NULL || gbdt_single_tree == NULL) 
		return -1;

 	int k = 0;
 	while (gbdt_single_tree->nodestatus[k] != GBDT_TERMINAL) 
 	{
		/* go down the tree */
		int m = gbdt_single_tree->splitid[k];
 		if (x_test[m] <= gbdt_single_tree->splitvalue[k]) 
			k = gbdt_single_tree->lson[k]; 
 		else 
			k = gbdt_single_tree->rson[k]; 
 	} 
 	ypred += shrink * gbdt_single_tree->ndavg[k];   // 预测结果

 	return 0;
}
  
/*************************************************************************
* training function
*************************************************************************/

int print_usage (FILE* stream, char* program_name)
{
	fprintf (stream, "Usage: %s options [ inputfile ... ]\n", program_name);
 	fprintf (stream, " -h --help.\n"
 	" -p --sample_ratio sample ratio learning rate, default is 0.5.\n"
 	" -r --sample_feature_ratio Feature ratio learning rate, default is 1.\n"
 	" -t --tree_num GBDT #trees in the model,default is 100\n"
 	" -s --shrink : learning rate, default is 0.03\n"
 	" -n --min_node_size : stop tree growing, minimum samples in leaf node, default is 30.\n"
 	" -d --max_depth stop tree growing, maximum depth of the tree, default is 5.\n"
 	" -m --model_out,default is out.model.\n"
 	" -f --train_file,default is train .\n"
 	);

 	return 0;
}
  
int init_info(gbdt_info_t& infbox)
{
	infbox.data_num = -1;
 	infbox.fea_num = -1;
 	infbox.gbdt_max_depth = 5;
 	infbox.gbdt_min_node_size = 30;
 	strcpy(infbox.model_filename, "out.model");
 	strcpy(infbox.train_filename, "train");
 	infbox.rand_fea_num = -1;
 	infbox.sample_num = -1;
 	infbox.shrink = 0.03;
 	infbox.tree_num = 100;

 	return 0;
}

int read_conf_file(gbdt_info_t& infbox, int argc, char* argv[])
{
	if(argv == NULL)	{
		LOG_ERROR_("Parameter error.");
		return -1;
	}

	if(argc < 2)	{
		print_usage(stderr, argv[0]);
		return -1;
	}

	int ch;
	double sample_ratio = 0.5;
	double random_feature_ratio = 1;
	char message[BUFFER_LENGTH];

	const char* short_options = "h:p:r:t:s:n:d:m:f:";

	const struct option long_options[]={
 	{"help", 0, NULL, 'h'},
 	{"sample_ratio", 0, NULL, 'p'},
 	{"sample_feature_ratio", 0, NULL, 'r'},
 	{"tree_num", 0, NULL, 't'},
 	{"shrink", 0, NULL, 's'},
 	{"min_node_size", 0, NULL, 'n'},
 	{"max_depth", 0, NULL, 'd'},
 	{"model_out", 0, NULL, 'm'},
 	{"train_file", 0, NULL, 'f'},
 	{NULL, 0, NULL, 0}};

 	init_info(infbox);

 	while((ch = getopt_long (argc, argv, short_options, long_options, NULL)) != -1)
 	{
		switch(ch)
 		{
			case 'h':
				if(argc == 2)
				{
					print_usage(stderr, argv[0]);
					return 1;
				}
				else
					return -1;
			case 'p':
				if(sscanf(optarg, "%lf", &sample_ratio) != 1)
 				{
					LOG_ERROR_("Get sample_ratio config error.");
					return -1;
 				}
 				break;
			case 'r':
				if(sscanf(optarg, "%lf", &random_feature_ratio) != 1)
 				{
					LOG_ERROR_("Get random_feature_ratio config error.");
					return -1;
 				}
 				break;
			case 't':
				if(sscanf(optarg, "%d", &infbox.tree_num) != 1)
 				{
					LOG_ERROR_("Get tree_num config error.");
					return -1;
 				}
				break;
 			case 's':
				if(sscanf(optarg, "%lf", &infbox.shrink) != 1)
 				{
					LOG_ERROR_("Get shrink config error.");
					return -1;
 				}
 				break;
			case 'n':
 				if(sscanf(optarg, "%d", &infbox.gbdt_min_node_size) != 1)
 				{
					LOG_ERROR_("Get min_node_size config error.");
					return -1;
				}
				break;
 			case 'd':
 				if(sscanf(optarg, "%d", &infbox.gbdt_max_depth) != 1)
 				{
					LOG_ERROR_("Get max_depth config error.");
					return -1;
 				}
 				break;
 			case 'm':
 				if(strlen(optarg) <= BUFFER_LENGTH)
					strncpy(infbox.model_filename, optarg, BUFFER_LENGTH);
 				else
 				{
					LOG_ERROR_("Get model_filename config error.");
					return -1;
 				}
 				break;
 			case 'f':
 				if(strlen(optarg) <= BUFFER_LENGTH)
					strncpy(infbox.train_filename, optarg, BUFFER_LENGTH);
 				else
 				{
					LOG_ERROR_("Get train_filename config error.");
					return -1;
 				}
 				break;
 			case '?':
 				print_usage(stderr, argv[0]);
 				return -1;
 			case -1:
 				print_usage(stderr, argv[0]);
 				return -1;
 			default:
 				print_usage(stderr, argv[0]);
 				return -1;
 		}
 	}

 	ifstream in(infbox.train_filename);

 	if(in)
 	{
		string line;
		infbox.data_num = 0;
 		infbox.fea_num = 0;
 		int temp;

 		while(getline(in, line))
 		{
			temp = max_feature_label(line);
			if(infbox.fea_num < temp)
			{
				infbox.fea_num = temp;
			}
			infbox.data_num++;
 		}
 		in.close();
 	}
 	else
 	{
		LOG_ERROR_("Can't open the train data file.");
		return -1;
 	}

 	snprintf(message, BUFFER_LENGTH, "Data Num: %d", infbox.data_num);
 	LOG_NOTICE_(message);

 	infbox.fea_num++; //从0开始。。。
 	snprintf(message, BUFFER_LENGTH, "Feature Num: %d", infbox.fea_num);
 	LOG_NOTICE_(message);

 	infbox.sample_num = (int)(infbox.data_num * sample_ratio);
 	infbox.rand_fea_num = (int)(infbox.fea_num * random_feature_ratio);

 	if( infbox.data_num <=0 || 
		infbox.fea_num <= 0 || 
		infbox.gbdt_max_depth <= 0 || 
		infbox.gbdt_min_node_size <= 0 || 
		infbox.rand_fea_num <= 0 || 
		infbox.rand_fea_num > infbox.fea_num || 
		infbox.sample_num <= 0 || 
		infbox.shrink <=0 || 
		infbox.shrink > 1 || 
		infbox.tree_num <= 0)
 	{
        
		return -1;
 	}

 	return 0;
}



int fill_novalue_feature(double* x, int fea_num, int data_num, double* faverage)
{
	if(x == NULL || faverage == NULL)
 	{
		printf("Parameter error.");
		return -1;
 	}
 	double sum = 0, avg;
 	int cnt = 0;
 	int index;
 	int novalue_n;
 	vector<int> novalues;

 	for(int i = 0; i < fea_num; i++)
 	{
		sum = 0;
 		cnt = 0;
 		novalue_n = 0;

 		novalues.clear();
 		for(int j = 0; j < data_num; j++)
 		{
			index = j*fea_num + i;
			if(x[index] != NO_VALUE)
			{
				sum += x[index];
				cnt++;
			}
			else
			{
				novalues.push_back(index);
			}
		}

		if(cnt > 0)
		{
			avg = sum / (double)cnt;
			faverage[i] = avg;
			novalue_n = novalues.size();

			for(int j = 0; j < novalue_n; j++)
			{
				x[novalues[j]] = avg;
			}
		}
 	}

 	return 0;
}


//
//  gbdt_regression_train main routine;
//     parameter :
//			x_fea_value : 2-dim matrix ;  sample x feature 
//			y_result_score : 
//			infbox:  model training configuration parameters; 
//
gbdt_model_t* gbdt_regression_train(double *x_fea_value, double *y_result_score, gbdt_info_t infbox)
{
 	bool failed = false;

 	gbdt_model_t* gbdt_model = (gbdt_model_t*)calloc(1, sizeof(gbdt_model_t));

 	gbdt_model->info = infbox;
 	gbdt_model->feature_average = (double*)calloc(gbdt_model->info.fea_num, sizeof(double));   // 每个feature算一个均值
 	gbdt_model->reg_forest = (gbdt_tree_t**) calloc(gbdt_model->info.tree_num, sizeof(gbdt_tree_t*)); // 森林，树的数目
 	// 计算gbdt_model->feature_average，每个特征的均值，对于x中的有些样本不含有某个特征，则赋予均值
 	fill_novalue_feature(x_fea_value, gbdt_model->info.fea_num, gbdt_model->info.data_num, gbdt_model->feature_average);

 	int nrnodes = 2* infbox.sample_num +1; // 3001 = 1500*2 +1 

 	// gbdt_inf 
 	//int* sample_in = (int *) calloc(infbox.data_num, sizeof(int));
 	double* y_select  = (double *) calloc(infbox.sample_num, sizeof(double));
 	double* x_select = (double *) calloc(infbox.fea_num * infbox.sample_num, sizeof(double));
 	int* index = (int *) calloc(infbox.sample_num, sizeof(int));
 	bufset* data_set = (bufset*)calloc(1, sizeof(bufset));
 	gbdt_tree_t* pgbdtree = (gbdt_tree_t*) calloc (1, sizeof(gbdt_tree_t));

 	double* y_gradient  = (double *) calloc(infbox.data_num, sizeof(double));
 	double* y_pred = (double *) calloc(infbox.data_num, sizeof(double));
 	double* x_test = (double*) malloc (infbox.fea_num * sizeof(double));
 	
 	if(index == NULL || y_select == NULL || x_select == NULL 
 	|| data_set == NULL || pgbdtree == NULL || y_gradient == NULL || y_pred == NULL 
 	|| x_test == NULL)
 	{
		failed = true;
		LOG_ERROR_("Failed to allocate memory.");
		goto ROOT_EXIT;
 	}
 	

 	for (int i=0; i< infbox.data_num; i++) {
		y_gradient[i] = y_result_score[i]; // 标注的目标值
		y_pred[i] = 0;  // 训练过程中的预测值
 	}
    // 训练过程中用到的变量， e.g. infbox.sample_num=1500
 	data_set->fea_pool     = (int *) calloc(infbox.fea_num, sizeof(int));
 	data_set->fvalue_list  = (double *) calloc(infbox.sample_num, sizeof(double));
 	data_set->y_list       = (double *) calloc(infbox.sample_num, sizeof(double));
 	data_set->fv           = (double *) calloc(infbox.sample_num, sizeof(double));
 	data_set->order_i      = (int *) calloc(infbox.sample_num, sizeof(int));
    // e.g. nrnodes=3001，pgbdtree表示模型中的一棵树
 	pgbdtree->nodestatus   = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->depth        = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->ndstart      = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->ndcount      = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->lson         = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->rson         = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->splitid      = (int*) calloc (nrnodes, sizeof(int));
 	pgbdtree->splitvalue   = (double*) calloc (nrnodes, sizeof(double));
 	pgbdtree->ndavg        = (double*) calloc (nrnodes, sizeof(double));
 	pgbdtree->nodesize     = 0;

 	if(pgbdtree->nodestatus == NULL || pgbdtree->depth == NULL || pgbdtree->ndstart == NULL 
 	|| pgbdtree->ndcount == NULL || pgbdtree->lson == NULL || pgbdtree->rson == NULL 
 	|| pgbdtree->splitid == NULL || pgbdtree->splitvalue == NULL || pgbdtree->ndavg == NULL 
 	|| data_set->fea_pool == NULL || data_set->fvalue_list == NULL || data_set->y_list == NULL 
 	|| data_set->fv == NULL || data_set->order_i == NULL)
 	{
		failed = true;
 		LOG_ERROR_("Failed to allocate memory.");
 		goto EXIT;
 	}

 	srand((unsigned)time(NULL));
 	// training iteration， the main part
 	for (int j = 0; j < infbox.tree_num; ++j) // 
 	{
        printf("tree: %d\n",j);
 		//for (int i = 0; i< infbox.sample_num; i++) 	// 每棵树训练所用到的样本数e.g. 1500
		//	index[i]	 = i; // 节点开始的样本下表，index中是1,2,...，对应到样本空间的下标可能是乱序的
		/*for (int i = 0; i< infbox.data_num; i++) 	// 所有样本数 
			sample_in[i] = 0;
		//采样。。。
		int sample_count =0;
		while(sample_count < infbox.sample_num) {
			int idx =  rand() % infbox.data_num;
			if (sample_in[idx] == 0) {
				sample_in[i] = 1;
				index[sample_count] = idx;
				sample_count++; 
			}
		}*/
		if (infbox.sample_num < infbox.data_num) {
            //knuth sample
            int t = infbox.sample_num;
            int mod = 0;
            for (int k=0; k<infbox.data_num; k++) {
                mod = infbox.data_num - t;
                if (rand()%mod < t) {
                    index[infbox.sample_num-t] = k; 
                    t--;
                }
            }
		} else {
         for (int i = 0; i< infbox.sample_num; i++) // 每棵树训练所用到的样本数e.g. 1500
			index[i] = i;
		}
		
		pgbdtree->nodesize = 0; // clear pgbtree

		/* build a single regression tree */
		int ret = gbdt_single_tree_estimation(x_fea_value, y_gradient, infbox, data_set, index, pgbdtree, nrnodes);
 		if(ret != 0) {
            LOG_ERROR_("Training model failed.");
			goto EXIT;
 		}

		// copy pgbtree to gbdt_tree[j]
 		int ndsize = pgbdtree->nodesize;
 		gbdt_model->reg_forest[j] = (gbdt_tree_t*)calloc(1, sizeof(gbdt_tree_t));
		
		gbdt_model->reg_forest[j]->nodestatus = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->ndstart = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->ndcount = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->lson = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->rson = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->splitid = (int*) malloc (ndsize * sizeof(int));
 		gbdt_model->reg_forest[j]->splitvalue = (double*) malloc (ndsize * sizeof(double));
 		gbdt_model->reg_forest[j]->ndavg = (double*) malloc (ndsize * sizeof(double));
 		gbdt_model->reg_forest[j]->nodesize = ndsize;
 		
 		if(gbdt_model->reg_forest[j]->nodestatus == NULL || gbdt_model->reg_forest[j]->ndstart == NULL 
 		|| gbdt_model->reg_forest[j]->ndcount == NULL || gbdt_model->reg_forest[j]->lson == NULL 
 		|| gbdt_model->reg_forest[j]->rson == NULL || gbdt_model->reg_forest[j]->splitid == NULL 
 		|| gbdt_model->reg_forest[j]->splitvalue == NULL || gbdt_model->reg_forest[j]->ndavg == NULL)
		{
			LOG_ERROR_("Failed to allocate memory.");
			goto EXIT;
 		}
 		
        // 在模型里面存储当前生成的这棵树
 		for (int i=0; i<ndsize; i++) {
			gbdt_model->reg_forest[j]->nodestatus[i] = pgbdtree->nodestatus[i];
 			gbdt_model->reg_forest[j]->ndstart[i]	 = pgbdtree->ndstart[i];
 			gbdt_model->reg_forest[j]->ndcount[i]	 = pgbdtree->ndcount[i];
 			gbdt_model->reg_forest[j]->lson[i]		 = pgbdtree->lson[i];
 			gbdt_model->reg_forest[j]->rson[i]		 = pgbdtree->rson[i];
 			gbdt_model->reg_forest[j]->splitid[i]	 = pgbdtree->splitid[i];
 			gbdt_model->reg_forest[j]->splitvalue[i] = pgbdtree->splitvalue[i];
 			gbdt_model->reg_forest[j]->ndavg[i]		 = pgbdtree->ndavg[i];
		}
		
        // 每次训练为止的效果评估
 		// 更新所有样本的y_gradient
 		for (int i=0; i< infbox.data_num; i++)
 		{
			for (int k=0; k<infbox.fea_num; k++)
 			{
				x_test[k] = x_fea_value[i * infbox.fea_num + k];
 			}
 			// 预测当前训练样本
 			gbdt_tree_predict(x_test, gbdt_model->reg_forest[j], y_pred[i], infbox.shrink);
 			y_gradient[i] = y_result_score[i] - y_pred[i]; // 更新梯度
		}
 	}
 	/* ===== end of tree iterations =====*/
EXIT:
 	if(data_set->fea_pool != NULL)	free(data_set->fea_pool);
 	if(data_set->fvalue_list != NULL)	free(data_set->fvalue_list);
 	if(data_set->y_list != NULL)	free(data_set->y_list);
 	if(data_set->fv != NULL)	free(data_set->fv);
 	if(data_set->order_i != NULL) 	free(data_set->order_i);

 	if(pgbdtree->nodestatus != NULL)	free(pgbdtree->nodestatus);
 	if(pgbdtree->depth != NULL)	free(pgbdtree->depth);
 	if(pgbdtree->ndstart != NULL)	free(pgbdtree->ndstart);
 	if(pgbdtree->ndcount != NULL)	free(pgbdtree->ndcount);
 	if(pgbdtree->lson != NULL)	free(pgbdtree->lson);
 	if(pgbdtree->rson != NULL)	free(pgbdtree->rson);
 	if(pgbdtree->splitid != NULL)	free(pgbdtree->splitid);
 	if(pgbdtree->splitvalue != NULL)	free(pgbdtree->splitvalue);
 	if(pgbdtree->ndavg != NULL)	 	free(pgbdtree->ndavg);

ROOT_EXIT:
 	//if(sample_in != NULL)	free(sample_in);
 	if(index != NULL)	 	free(index);
 	if(y_select != NULL) 	free(y_select);
 	if(x_select != NULL) 	free(x_select);
 	if(data_set != NULL) 	free(data_set);
 	if(pgbdtree != NULL) 	free(pgbdtree);

 	if(y_gradient != NULL) 	free(y_gradient);
 	if(y_pred != NULL)	 	free(y_pred);
 	if(x_test != NULL)	 	free(x_test);

 	if(!failed) 	{
	 	return gbdt_model;
 	}
 	else 	{
	 	free_model(gbdt_model);
		return NULL;
 	}
}


int save_gbdt_info(gbdt_info_t infbox, FILE* model_fp)
{
 	fwrite(&infbox.tree_num, sizeof(int), 1, model_fp) ;
 	fwrite(&infbox.fea_num, sizeof(int), 1, model_fp) ;
 	fwrite(&infbox.data_num, sizeof(int), 1, model_fp) ;
	fwrite(&infbox.rand_fea_num, sizeof(int), 1, model_fp) ; 
 	fwrite(&infbox.shrink, sizeof(double), 1, model_fp) ;
 	fwrite(&infbox.gbdt_min_node_size, sizeof(int), 1, model_fp) ; 
 	fwrite(&infbox.gbdt_max_depth, sizeof(int), 1, model_fp) ;
 	
 	return 0;
}
int gbdt_save_reg_forest(FILE* model_fp, gbdt_tree_t** reg_forest, gbdt_info_t infbox)
{
 	int nodesize;

 	for(int i = 0; i < infbox.tree_num; i++)
 	{
		nodesize = reg_forest[i]->nodesize;
 		fwrite(&nodesize, sizeof(int), 1, model_fp) ;
		fwrite(reg_forest[i]->nodestatus, sizeof(int), nodesize, model_fp) ;
 		fwrite(reg_forest[i]->splitid, sizeof(int), nodesize, model_fp) ;
 		fwrite(reg_forest[i]->splitvalue, sizeof(double), nodesize, model_fp) ;
 		fwrite(reg_forest[i]->ndavg, sizeof(double), nodesize, model_fp) ; 
		fwrite(reg_forest[i]->rson, sizeof(int), nodesize, model_fp) ; 
 		fwrite(reg_forest[i]->lson, sizeof(int), nodesize, model_fp) ;
 	}
 	return 0;
}


int gbdt_save_model(gbdt_model_t* gbdt_model, char* model_filename)
{
 	FILE* model_fp = fopen(model_filename, "wb");

 	save_gbdt_info(gbdt_model->info, model_fp) ; 

 	gbdt_save_reg_forest(model_fp, gbdt_model->reg_forest, gbdt_model->info); 

 	if(gbdt_model->feature_average != NULL) {
		fwrite(gbdt_model->feature_average, sizeof(double), gbdt_model->info.fea_num, model_fp);
 	}

	fclose(model_fp);

	return 0;
}
   
   
int gbdt_load_reg_forest(FILE* model_fp, gbdt_model_t* gbdt_model)
{
	gbdt_model->reg_forest = (gbdt_tree_t**) malloc(gbdt_model->info.tree_num * sizeof(gbdt_tree_t*));

 	gbdt_tree_t* prtree;
 	int nodesize, rsize;

	for(int i = 0; i < gbdt_model->info.tree_num; i++)
 	{
		rsize = fread(&nodesize, sizeof(int), 1, model_fp);

		gbdt_model->reg_forest[i] = (gbdt_tree_t*) calloc (1, sizeof(gbdt_tree_t));
		prtree = gbdt_model->reg_forest[i];
		prtree->nodestatus = (int*) malloc (nodesize * sizeof(int));
 		prtree->lson = (int*) malloc (nodesize * sizeof(int));
		prtree->rson = (int*) malloc (nodesize * sizeof(int));
 		prtree->splitid = (int*) malloc (nodesize * sizeof(int));
 		prtree->splitvalue = (double*) malloc (nodesize * sizeof(double));
 		prtree->ndavg = (double*) malloc (nodesize * sizeof(double));
		prtree->nodesize = nodesize;

		fread(prtree->nodestatus, sizeof(int), nodesize, model_fp); 
 		fread(prtree->splitid, sizeof(int), nodesize, model_fp); 
 		fread(prtree->splitvalue, sizeof(double), nodesize, model_fp); 
 		fread(prtree->ndavg, sizeof(double), nodesize, model_fp);
 		fread(prtree->rson, sizeof(int), nodesize, model_fp); 
 		fread(prtree->lson, sizeof(int), nodesize, model_fp);

 		prtree->ndstart = NULL;
 		prtree->ndcount = NULL;
	}
	return 0;
}



int load_gbdt_info(gbdt_info_t* pinfbox, FILE* model_fp)
{
 	fread(&pinfbox->tree_num, sizeof(int), 1, model_fp) ;
 	fread(&pinfbox->fea_num, sizeof(int), 1, model_fp) ;
	fread(&pinfbox->data_num, sizeof(int), 1, model_fp) ;
 	fread(&pinfbox->rand_fea_num, sizeof(int), 1, model_fp) ;
 	fread(&pinfbox->shrink, sizeof(double), 1, model_fp) ;
 	fread(&pinfbox->gbdt_min_node_size, sizeof(int), 1, model_fp) ;
 	fread(&pinfbox->gbdt_max_depth, sizeof(int), 1, model_fp); 

 	return 0;
}


gbdt_model_t* gbdt_load_model(char* model_file)
{
 	FILE* model_fp = fopen(model_file, "rb");

	gbdt_model_t* gbdt_model =  (gbdt_model_t*)calloc(1, sizeof(gbdt_model_t));
 	load_gbdt_info(&gbdt_model->info, model_fp); 
	gbdt_load_reg_forest(model_fp, gbdt_model); 

	gbdt_model->feature_average = (double*)malloc(gbdt_model->info.fea_num * sizeof(double));
	fread(gbdt_model->feature_average, sizeof(double), gbdt_model->info.fea_num, model_fp);

 	fclose(model_fp);

	return gbdt_model;
}

int free_model(gbdt_model_t*& gbdt_model)
{
 gbdt_tree_t* prtree;

 if(gbdt_model->reg_forest != NULL)
 {
 for(int i = 0; i < gbdt_model->info.tree_num; i++)
 {
 prtree = gbdt_model->reg_forest[i];
 if(prtree != NULL)
 {
 if(prtree->nodestatus != NULL)
 {
 free(prtree->nodestatus);
 prtree->nodestatus = NULL;
 }
 if(prtree->depth != NULL)
 {
 free(prtree->depth);
 prtree->depth = NULL;
 }
 if(prtree->lson != NULL)
 {
 free(prtree->lson);
 prtree->lson = NULL;
 }
 if(prtree->rson != NULL)
 {
 free(prtree->rson);
 prtree->rson = NULL;
 }
 if(prtree->splitid != NULL)
 {
 free(prtree->splitid);
 prtree->splitid = NULL;
 }
 if(prtree->splitvalue != NULL)
 {
 free(prtree->splitvalue);
 prtree->splitvalue = NULL;
 }
 if(prtree->ndavg != NULL)
 {
 free(prtree->ndavg);
 prtree->ndavg = NULL;
 }
 if(prtree->ndstart != NULL)
 {
 free(prtree->ndstart);
 prtree->ndstart = NULL;
 }
 if(prtree->ndcount != NULL)
 {
 free(prtree->ndcount);
 prtree->ndcount = NULL;
 }
 free(gbdt_model->reg_forest[i]);
 gbdt_model->reg_forest[i] = NULL;
 }
 }
 free(gbdt_model->reg_forest);
 gbdt_model->reg_forest = NULL;
 }

 if(gbdt_model->feature_average != NULL)
 {
 free(gbdt_model->feature_average);
 gbdt_model->feature_average = NULL;
 }

 free(gbdt_model);
 gbdt_model = NULL;

 return 1;
}
   
#define qsort_Index
#define NUMERIC double
void R_qsort_I(double *v, int *I, int i, int j)
{
	/* Orders v[] increasingly. Puts into I[] the permutation vector:
 	*  new v[k] = old v[I[k]]
 	* Only elements [i : j]  (in 1-indexing !)  are considered.
 	*/

 	int il[31], iu[31];
 	NUMERIC vt, vtt;
 	double R = 0.375;  // 设置默认值，用于选择作为比较标准的样本点
 	int ii, ij, k, l, m;
#ifdef qsort_Index
	int it, tt;
#endif


	/* 1-indexing for I[], v[]  (and `i' and `j') : */
	--v;
#ifdef qsort_Index
	--I;
#endif

	ii = i;/* save */
 	m = 1;

L10:
 if (i < j) {
 if (R < 0.5898437) R += 0.0390625; else R -= 0.21875;
L20:
 k = i;
 /* ij = (j + i) >> 1; midpoint */
 ij = i + (int)((j - i)*R);   // 计算选定的标准样本点下标
#ifdef qsort_Index
 it = I[ij];
#endif
 // 通过比较进行交换，标准样本之前的样本小于其，之后的大于其
 
 vt = v[ij];
 if (v[i] > vt) {
#ifdef qsort_Index
 I[ij] = I[i]; I[i] = it; it = I[ij];
#endif
 v[ij] = v[i]; v[i] = vt; vt = v[ij];
 }
 /* L30:*/
 l = j;
 if (v[j] < vt) {
#ifdef qsort_Index
 I[ij] = I[j]; I[j] = it; it = I[ij];
#endif
 v[ij] = v[j]; v[j] = vt; vt = v[ij];
 if (v[i] > vt) {
#ifdef qsort_Index
 I[ij] = I[i]; I[i] = it; it = I[ij];
#endif
 v[ij] = v[i]; v[i] = vt; vt = v[ij];
 }
 }

 for(;;) { /*L50:*/   // vt分割，之前的小于其，之后的大于其
 //do l--;  while (v[l] > vt);
 l--;for(;v[l]>vt;l--);


#ifdef qsort_Index
 tt = I[l];
#endif
 vtt = v[l];
 /*L60:*/ 
 //do k++;  while (v[k] < vt);
 k=k+1;for(;v[k]<vt;k++);

 if (k > l) break;

 /* else (k <= l) : */
#ifdef qsort_Index
 I[l] = I[k]; I[k] =  tt;
#endif
 v[l] = v[k]; v[k] = vtt;
 }

 m++;
 if (l - i <= j - k) {
 /*L70: */
 il[m] = k;
 iu[m] = j;
 j = l;
 }
 else {
 il[m] = i;
 iu[m] = l;
 i = k;
 }
 }else { /* i >= j : */

L80:
 if (m == 1) return;

 /* else */
 i = il[m];
 j = iu[m];
 m--;
 }

 if (j - i > 10)  goto L20;

 if (i == ii)  goto L10;

 --i;
L100:
 do {
 ++i;
 if (i == j) {
 goto L80;
 }
#ifdef qsort_Index
 it = I[i + 1];
#endif
 vt = v[i + 1];
 } while (v[i] <= vt);

 k = i;

 do { /*L110:*/
#ifdef qsort_Index
 I[k + 1] = I[k];
#endif
 v[k + 1] = v[k];
 --k;
 } while (vt < v[k]);

#ifdef qsort_Index
 I[k + 1] = it;
#endif
 v[k + 1] = vt;
 goto L100;
}  // end of  R_qsort_I(double *v, int *I, int i, int j)， 实现按选定特征值的升序排序
   
   
   
   
   
   
   
   
   
   
