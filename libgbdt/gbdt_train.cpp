#include "gradient_boosting.h"

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

void my_memset(double* x, int count, int value)
{
	for(int i = 0; i < count; i++)
	{
		x[i] = value;
	}
}
 
 bool has_colon(string item)
 {
	for(int i = 0; i < (int)item.size(); i++)
  	{
		if(item[i] == ':')
		return true;
  	}
 
  	return false;
 }
 
int read_train_file(double*& x, double*& y, gbdt_info_t infbox)
{
	double value;

	x = (double*) malloc (infbox.data_num * infbox.fea_num * sizeof(double));  // 样本数*特征数 训练样本特征
	if(x == NULL)
	{
		LOG_ERROR_("Failed to allocate memory.");
		return -1;
	}
	y = (double*) malloc (infbox.data_num * sizeof(double)); // 样本数, 存储样本标注值
	if(y == NULL)
	{
		LOG_ERROR_("Failed to allocate memory.");
		free(x);
		return -1;
	}

	my_memset(x, infbox.data_num * infbox.fea_num, NO_VALUE);
	my_memset(y, infbox.data_num, NO_VALUE);

	int cnt = 0;
	int count = 0;
	int fid;

	ifstream fptrain(infbox.train_filename); // 训练数据文件
	string* items = new string[infbox.fea_num+5];
	if(items == NULL)
	{
		LOG_ERROR_("Failed to allocate memory.");
		free(x);
		free(y);
		fptrain.close();
		return -1;
	}
	string line;
	int x_read;

	while (getline(fptrain, line) != NULL)
	{
		count = splitline(line, items, infbox.fea_num+5, ' '); // 以空格切割样本的每个特征，返回特征数

		if(count < 2 || count > infbox.fea_num+5)
 		{
			delete[] items;
 			free(x);
 			free(y);
 			fptrain.close();
 			LOG_ERROR_("Read train data error");
 			return -1;
 		}

		if(sscanf(items[0].c_str(),"%lf",&value) != 1)
 		{
			delete[] items;
			free(x);
			free(y);
			fptrain.close();
			LOG_ERROR_("Read train data error");
			return -1;
 		}
		y[cnt] = value; //the first column is y,  训练目标值，cnt表示样本计数下标

		for(int i = 1; i < count; i++)
		{
			if(has_colon(items[i]))
			{
				x_read = sscanf(items[i].c_str(),"%d:%lf", &fid, &value); //featureid1:value1 featureid2:value2 ... density matrix x

				if (fid >= infbox.fea_num || x_read != 2)
				{
					delete[] items;
					free(x);
					free(y);
					fptrain.close();
					LOG_ERROR_("Read feature error");
					return -1;
				}
				x[cnt*infbox.fea_num + fid] = value; // 每个样本(cnt)的特征(fid)对应的特征值(value)，没有则是默认值NO_VALUE
			}
		}
		cnt ++;
		if (cnt >= infbox.data_num)
		{
			break;
		}
	}
 
	delete[] items;
	fptrain.close();
	return 0;
}
 
int main(int argc, char* argv[])
{
	double *x = NULL;
	double *y = NULL;
	gbdt_info_t infbox;

	char log_message[BUFFER_LENGTH];

	LOG_NOTICE_("Reading config ... ");

	int res = read_conf_file(infbox, argc, argv); // 用默认值初始化配置参数
    /*
    1. infbox.train_filename指定用于模型训练的文件，e.g. 每一条样本(一行)为 1 0:26503 1:511405 2:511405 3:500000 4:366 5:19.2954
    2. infbox.fea_num表示训练文件中的特征数目(e.g. 6)，如上样本示例，特征从0开始计数
    3. infbox.data_num表示样本数(e.g. 1500)，即infbox.train_filename文件的行数
    4. infbox.rand_fea_num表示每棵树训练时，随机选择的特征数(e.g. 4)，每选择一个特征，计算一次分裂损失函数，对比得到最优的分裂特征
    5. infbox.sample_num表示训练每一棵树需要的样本数(e.g. 1500)
    6. infbox.gbdt_max_depth表示树的深度(e.g. 5)
    */
	if(res == -1)
	{
		LOG_ERROR_("Read parameter failed.");
		return -1;
	}
	else if(res == 1)
	{
		return 0;
	}

	snprintf(log_message, BUFFER_LENGTH, "Reading training data from: %s ...", infbox.train_filename);
	LOG_NOTICE_(log_message);

	if(read_train_file(x, y, infbox) != 0)  // 读入训练文件
	{
		LOG_ERROR_("Failed to read training data file.");
		return -1;
	}

	LOG_NOTICE_("Training...");

	gbdt_model_t* gbdt_model = gbdt_regression_train(x, y, infbox);  // 模型训练
	if(gbdt_model == NULL) // gbdt_model表示训练得到的最终的模型，其具体由gbdt_model->reg_forest指代
	{
		LOG_ERROR_("Training Model Failed.");
		return -1;
	}

	LOG_NOTICE_("Saving Model ... ");
	// 存储模型参数
	if(gbdt_save_model(gbdt_model, infbox.model_filename) != 0)
	{
		LOG_ERROR_("Saving Model Failed.");
		return -1;
	}
	LOG_NOTICE_("Done.");

	free_model(gbdt_model);
	if(x != NULL)
	{
		free(x);
	}
	if(y != NULL)
	{
		free(y);
	}

	return 0;
}
