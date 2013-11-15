#include "gradient_boosting.h"

#include <iostream>
#include <fstream>
#include <string>
using namespace std;


bool has_colon(string item)
 {
  for(int i = 0; i < (int)item.size(); i++)
  {
  if(item[i] == ':')
  {
  return true;
  }
  }
 
  return false;
 }
 
void fill_no_value_aver(gbdt_model_t* gbdt_model, double* x_test)
{
 for(int i = 0; i < gbdt_model->info.fea_num; i++)
 {
 if(x_test[i] == NO_VALUE)
 {
 x_test[i] = gbdt_model->feature_average[i];
 }
 }
}

int main(int argc, char* argv[])
{
	if (argc != 2)
 	{
		fprintf(stderr, "Usage: %s model_filename <stdin> <stdout>\n", argv[0]);
		return -1;
 	}

 	char* model_filename = argv[1];

 	gbdt_model_t* gbdt_model = gbdt_load_model(model_filename);

 	if(gbdt_model == NULL)
 	{
		LOG_ERROR_("Failed to load GBDT model!\n");
		return -1;
 	}

	int feat;
 	double *x_test = (double*) malloc(gbdt_model->info.fea_num * sizeof(double));
 	if(x_test == NULL)
 	{
		LOG_ERROR_("Failed to allocate memory.");
 		free_model(gbdt_model);
 		return -1;
 	}

	double reg_predict = 0;
 	double value = 0;
 	double realv = 0;

 	int fea_count;

 	string line;
 	string* items = new string[gbdt_model->info.fea_num+5];

 	if(items == NULL)
 	{
		LOG_ERROR_("Failed to allocate memory.");
 		free(x_test);
 		free_model(gbdt_model);
 		return -1;
 	}
	while (getline(cin, line))
 	{
		for(int j = 0; j < gbdt_model->info.fea_num; j++)
			x_test[j] = NO_VALUE;

 		fea_count=splitline(line, items, gbdt_model->info.fea_num+3, ' ');
 		if(fea_count < 2 || fea_count > gbdt_model->info.fea_num+3)
 		{
			LOG_ERROR_("Test Data Format Error.");
 			delete[] items;
 			free(x_test);
 			free_model(gbdt_model);
 			return -1;
 		}
		if(sscanf(items[0].c_str(),"%lf",&realv) != 1)
 		{
			LOG_ERROR_("Failed to read feature data.");
 			delete[] items;
 			free(x_test);
 			free_model(gbdt_model);
 			return -1;
 		}

		for(int i = 1; i < fea_count; i++)
  		{
			if(has_colon(items[i]))
  			{
				if(sscanf(items[i].c_str(),"%d:%lf", &feat, &value) != 2)
  				{
					LOG_ERROR_("Failed to read feature data.");
  					delete[] items;
  					free(x_test);
  					free_model(gbdt_model);
  					return -1;
				}
				if(feat < 0 || feat >= gbdt_model->info.fea_num)
  				{
					LOG_ERROR_("Failed to read feature data.");
  					delete[] items;
  					free(x_test);
  					free_model(gbdt_model);
  					return -1;
  				}
  				x_test[feat] = value;
  			}
		}
		fill_no_value_aver(gbdt_model, x_test);
 
  		if(gbdt_regression_predict(gbdt_model, x_test, reg_predict) == 0)
  		{
			printf("%d\t%lf\n", (int)realv, reg_predict);
  		}
	}
 
	delete[] items;
 
  	free(x_test);
 
  	free_model(gbdt_model);
 
  	return 0;
}
