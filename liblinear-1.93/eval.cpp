#include <iostream>
#include <vector>
#include <algorithm>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include <math.h>
#include "linear.h"
#include "eval.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef std::vector<double> dvec_t;
typedef std::vector<int> ivec_t;

// prototypes of evaluation functions
double precision(const dvec_t& dec_values, const ivec_t& ty);
double recall(const dvec_t& dec_values, const ivec_t& ty);
double fscore(const dvec_t& dec_values, const ivec_t& ty);
double bac(const dvec_t& dec_values, const ivec_t& ty);
double auc(const dvec_t& dec_values, const ivec_t& ty);
double auc2(const dvec_t& dec_values, const ivec_t& ty);
double auc_ctr(const dvec_t& dec_values, const ivec_t& clicks,
		const ivec_t& shows);
double accuracy(const dvec_t& dec_values, const ivec_t& ty);

// evaluation function pointer
// You can assign this pointer to any above prototype
double (*validation_function)(const dvec_t&, const ivec_t&) = auc2;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input) {
	int len;

	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(line, '\n') == NULL) {
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

double precision(const dvec_t& dec_values, const ivec_t& ty) {
	size_t size = dec_values.size();
	size_t i;
	int tp, fp;
	double precision;

	tp = fp = 0;

	for (i = 0; i < size; ++i)
		if (dec_values[i] >= 0) {
			if (ty[i] == 1)
				++tp;
			else
				++fp;
		}

	if (tp + fp == 0) {
		fprintf(stderr, "warning: No postive predict label.\n");
		precision = 0;
	} else
		precision = tp / (double) (tp + fp);
	printf("Precision = %g%% (%d/%d)\n", 100.0 * precision, tp, tp + fp);

	return precision;
}

double recall(const dvec_t& dec_values, const ivec_t& ty) {
	size_t size = dec_values.size();
	size_t i;
	int tp, fn; // true_positive and false_negative
	double recall;

	tp = fn = 0;

	for (i = 0; i < size; ++i)
		if (ty[i] == 1) { // true label is 1
			if (dec_values[i] >= 0)
				++tp; // predict label is 1
			else
				++fn; // predict label is -1
		}

	if (tp + fn == 0) {
		fprintf(stderr, "warning: No postive true label.\n");
		recall = 0;
	} else
		recall = tp / (double) (tp + fn);
	// print result in case of invocation in prediction
	printf("Recall = %g%% (%d/%d)\n", 100.0 * recall, tp, tp + fn);

	return recall; // return the evaluation value
}

double fscore(const dvec_t& dec_values, const ivec_t& ty) {
	size_t size = dec_values.size();
	size_t i;
	int tp, fp, fn;
	double precision, recall;
	double fscore;

	tp = fp = fn = 0;

	for (i = 0; i < size; ++i)
		if (dec_values[i] >= 0 && ty[i] == 1)
			++tp;
		else if (dec_values[i] >= 0 && ty[i] == -1)
			++fp;
		else if (dec_values[i] < 0 && ty[i] == 1)
			++fn;

	if (tp + fp == 0) {
		fprintf(stderr, "warning: No postive predict label.\n");
		precision = 0;
	} else
		precision = tp / (double) (tp + fp);
	if (tp + fn == 0) {
		fprintf(stderr, "warning: No postive true label.\n");
		recall = 0;
	} else
		recall = tp / (double) (tp + fn);

	if (precision + recall == 0) {
		fprintf(stderr, "warning: precision + recall = 0.\n");
		fscore = 0;
	} else
		fscore = 2 * precision * recall / (precision + recall);

	printf("F-score = %g\n", fscore);

	return fscore;
}

double bac(const dvec_t& dec_values, const ivec_t& ty) {
	size_t size = dec_values.size();
	size_t i;
	int tp, fp, fn, tn;
	double specificity, recall;
	double bac;

	tp = fp = fn = tn = 0;

	for (i = 0; i < size; ++i)
		if (dec_values[i] >= 0 && ty[i] == 1)
			++tp;
		else if (dec_values[i] >= 0 && ty[i] == -1)
			++fp;
		else if (dec_values[i] < 0 && ty[i] == 1)
			++fn;
		else
			++tn;

	if (tn + fp == 0) {
		fprintf(stderr, "warning: No negative true label.\n");
		specificity = 0;
	} else
		specificity = tn / (double) (tn + fp);
	if (tp + fn == 0) {
		fprintf(stderr, "warning: No positive true label.\n");
		recall = 0;
	} else
		recall = tp / (double) (tp + fn);

	bac = (specificity + recall) / 2;
	printf("BAC = %g\n", bac);

	return bac;
}

// only for auc
class Comp {
	const double *dec_val;
public:
	Comp(const double *ptr) :
			dec_val(ptr) {
	}
	bool operator()(int i, int j) const {
		return dec_val[i] > dec_val[j];
	}
};

//class Comp2 {
//	const double *dec_val;
//	const int *ty;
//public:
//	Comp2(const double *pdec_val, const int *pty) :
//			dec_val(pdec_val), ty(pty) {
//	}
//	bool operator()(int i, int j) const {
//		if (dec_val[i] > dec_val[j]) {
//			return true;
//		} else if (dec_val[i] == dec_val[j]) {
//			return ty[i] > ty[j];
//		}
//		return false;
//	}
//};

double auc(const dvec_t& dec_values, const ivec_t& ty) {
	double roc = 0;
	size_t size = dec_values.size();
	size_t i;
	std::vector<size_t> indices(size);

	for (i = 0; i < size; ++i)
		indices[i] = i;

	std::sort(indices.begin(), indices.end(), Comp(&dec_values[0]));

	int tp = 0, fp = 0;
	for (i = 0; i < size; i++) {
		if (ty[indices[i]] == 1) {
			tp++;
		} else if (ty[indices[i]] == -1) {
			roc += tp;
			fp++;
		}
	}

	if (tp == 0 || fp == 0) {
		fprintf(stderr,
				"warning: Too few postive true labels or negative true labels\n");
		roc = 0;
	} else
		roc = roc / tp / fp;

	printf("AUC = %g\n", roc);

	return roc;
}

double auc2(const dvec_t& dec_values, const ivec_t& ty) {
	double auc = 0;
	size_t size = dec_values.size();
	size_t i;
	std::vector<size_t> indices(size);

	for (i = 0; i < size; ++i) {
		indices[i] = i;
//		printf("%f %d\n", dec_values[i],ty[i]);
	}

	std::sort(indices.begin(), indices.end(), Comp(&dec_values[0]));
//	printf ("\n");
//	for (i = 0; i < size; ++i) {
//		printf("%d %f %d\n", indices[i], dec_values[indices[i]],ty[indices[i]]);
//	}

	int total_positive = 0;
	int total_negative = 0;

	for (i = 0; i < size; i++) {
		if (ty[i] > 0) {
			total_positive++;
		} else {
			total_negative++;
		}
	}

	double tp = 0;
	double fp = 0;
	double fp_prev = 0;
	double tp_prev = 0;
	double f_prev = -100000;

	for (i = 0; i < size; i++) {
		double cur_f = dec_values[indices[i]];
		if (cur_f != f_prev) {
			auc += fabs(fp - fp_prev) * ((tp + tp_prev) / 2);
			f_prev = cur_f;
			fp_prev = fp;
			tp_prev = tp;
		}

		if (ty[indices[i]] > 0) {
			tp++;
		} else {
			fp++;
		}
	}
	auc += fabs(total_negative - fp_prev) * ((total_positive + tp_prev) / 2.0);
	auc /= total_positive * total_negative;

	printf("AUC = %g\n", auc);

	return auc;
}

double auc_ctr(const dvec_t& predicted_ctr, const ivec_t& num_clicks,
		const ivec_t& num_impressions) {
	double auc = 0;
	size_t size = predicted_ctr.size();
	size_t i;
	std::vector<size_t> i_sorted(size);

	for (i = 0; i < size; ++i)
		i_sorted[i] = i;

	std::sort(i_sorted.begin(), i_sorted.end(), Comp(&predicted_ctr[0]));

	double auc_temp = 0;
	double click_sum = 0;
	double old_click_sum = 0;
	double no_click = 0;
	double no_click_sum = 0;
	double last_ctr = predicted_ctr[i_sorted[0]] + 1.0;
	for (i = 0; i < size; i++) {
		if (last_ctr != predicted_ctr[i_sorted[i]]) {
			auc_temp += (click_sum + old_click_sum) * no_click / 2.0;
			old_click_sum = click_sum;
			no_click = 0.0;
			last_ctr = predicted_ctr[i_sorted[i]];
		}
		no_click += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]];
		no_click_sum += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]];
		click_sum += num_clicks[i_sorted[i]];
	}
	auc_temp += (click_sum + old_click_sum) * no_click / 2.0;
	auc = auc_temp / (click_sum * no_click_sum);
	//printf("AUC = %g\n", auc);
	return auc;
}

double accuracy(const dvec_t& dec_values, const ivec_t& ty) {
	int correct = 0;
	int total = (int) ty.size();
	size_t i;

	for (i = 0; i < ty.size(); ++i)
		if (ty[i] == (dec_values[i] >= 0 ? 1 : -1))
			++correct;

	printf("Accuracy = %g%% (%d/%d)\n", (double) correct / total * 100, correct,
			total);

	return (double) correct / total;
}

double binary_class_cross_validation(const problem *prob,
		const parameter *param, int nr_fold) {
	printf("begin validtaion");
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);
	int *labels;
	dvec_t dec_values;
	ivec_t ty;

	for (i = 0; i < l; i++)
		perm[i] = i;
	for (i = 0; i < l; i++) {
		int j = i + rand() % (l - i);
		std::swap(perm[i], perm[j]);
	}
	for (i = 0; i <= nr_fold; i++)
		fold_start[i] = i * l / nr_fold;

	for (i = 0; i < nr_fold; i++) {
		int begin = fold_start[i];
		int end = fold_start[i + 1];
		int j, k;
		struct problem subprob;

		subprob.n = prob->n;
		subprob.bias = prob->bias;
		subprob.l = l - (end - begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k = 0;
		for (j = 0; j < begin; j++) {
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for (j = end; j < l; j++) {
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob, param);

		labels = Malloc(int, get_nr_class(submodel));
		get_labels(submodel, labels);

		if (get_nr_class(submodel) > 2) {
			fprintf(stderr, "Error: the number of class is not equal to 2\n");
			exit(-1);
		}

		dec_values.resize(end);
		ty.resize(end);

		for (j = begin; j < end; j++) {
			predict_values(submodel, prob->x[perm[j]], &dec_values[j]);
			ty[j] = (prob->y[perm[j]] > 0) ? 1 : -1;
		}

		if (labels[0] <= 0) {
			for (j = begin; j < end; j++)
				dec_values[j] *= -1;
		}

		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
		free(labels);
	}

	free(perm);
	free(fold_start);
	printf("end validtaion,begin auc");
	return validation_function(dec_values, ty);
}

double binary_class_cross_validation_ctr(const problem *prob,
		const sample *samp, const parameter *param, int nr_fold) {
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = samp->l;
	int *perm = Malloc(int,l);
	int *labels;

	dvec_t predicted_ctr;
	ivec_t num_clicks;
	ivec_t num_impressions;

	srand(time(NULL));
	for (i = 0; i < l; i++)
		perm[i] = i;
	for (i = 0; i < l; i++) {
		int j = i + rand() % (l - i);
		std::swap(perm[i], perm[j]); //
	}

//	for(i=0;i<l;i++) {
//		printf("%d ",perm[i]);
//	}
//	printf("\n");
//	printf("l:%d\n",l);
	for (i = 0; i <= nr_fold; i++) {
		fold_start[i] = i * l / nr_fold;
	}

	for (i = 0; i < nr_fold; i++) {
		int begin = fold_start[i];
		int end = fold_start[i + 1];
		int j, k;
		struct problem subprob;

//		printf("begin:%d,end:%d\n",begin,end);

		subprob.n = prob->n;
		subprob.bias = prob->bias;
		subprob.l = prob->l;
		for (j = begin; j < end; j++) {
			subprob.l -= samp->s[perm[j]].shows;
		}

//		printf("subprob.l:%d\n",subprob.l);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k = 0;
		for (j = 0; j < begin; j++) {
			for (int m = samp->s[perm[j]].start_index;
					m <= samp->s[perm[j]].end_index; m++) {
				subprob.x[k] = prob->x[m];
				subprob.y[k] = prob->y[m];
				++k;
			}
		}

		for (j = end; j < l; j++) {
			for (int m = samp->s[perm[j]].start_index;
					m <= samp->s[perm[j]].end_index; m++) {
				subprob.x[k] = prob->x[m];
				subprob.y[k] = prob->y[m];
				++k;
			}
		}
//		printf("k:%d\n",k);
		struct model *submodel = train(&subprob, param);

		int nr_class = get_nr_class(submodel);
		double *prob_estimates = Malloc(double,nr_class);

		labels = Malloc(int, nr_class);
		get_labels(submodel, labels);

		if (get_nr_class(submodel) > 2) {
			fprintf(stderr, "Error: the number of class is not equal to 2\n");
			exit(-1);
		}

		for (j = begin; j < end; j++) {
			int m = samp->s[perm[j]].start_index;
			predict_probability(submodel, prob->x[m], prob_estimates);
			double predict_ctr = prob_estimates[0];

			int clicks = samp->s[perm[j]].clicks;
			int shows = samp->s[perm[j]].shows;
			num_clicks.push_back(clicks);
			num_impressions.push_back(shows);

			predicted_ctr.push_back(predict_ctr);
		}

		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
		free(labels);
		free(prob_estimates);
	}

	free(perm);
	free(fold_start);

	double auc = auc_ctr(predicted_ctr, num_clicks, num_impressions);
	return auc;
}

void binary_class_predict(FILE *input, FILE *output) {
	int total = 0;
	int *labels;
	int max_nr_attr = 64;
	struct feature_node *x = Malloc(struct feature_node, max_nr_attr);
	dvec_t dec_values;
	ivec_t true_labels;
	int n;
	if (model_->bias >= 1)
		n = get_nr_feature(model_) + 1;
	else
		n = get_nr_feature(model_);

	labels = Malloc(int, get_nr_class(model_));
	get_labels(model_, labels);

	max_line_len = 1024;
	line = (char *) malloc(max_line_len * sizeof(char));
	while (readline(input) != NULL) {
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line, " \t");
		target_label = strtod(label, &endptr);
		if (endptr == label)
			exit_input_error(total + 1);

		while (1) {
			if (i >= max_nr_attr - 2)	// need one more for index = -1
					{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,
						max_nr_attr * sizeof(struct feature_node));
			}

			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0'
					|| x[i].index <= inst_max_index)
				exit_input_error(total + 1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val, &endptr);
			if (endptr == val || errno != 0
					|| (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total + 1);

			++i;
		}

		if (model_->bias >= 0) {
			x[i].index = n;
			x[i].value = model_->bias;
			++i;
		}

		x[i].index = -1;

		predict_label = predict(model_, x);
		fprintf(output, "%g\n", predict_label);

		double dec_value;
		predict_values(model_, x, &dec_value);
		true_labels.push_back((target_label > 0) ? 1 : -1);
		if (labels[0] <= 0)
			dec_value *= -1;
		dec_values.push_back(dec_value);
	}

	validation_function(dec_values, true_labels);

	free(labels);
	free(x);
}

void binary_class_predict_ctr(FILE *input, FILE *output) {
	int total = 0;
	int *labels;
	int max_nr_attr = 64;
	struct feature_node *x = Malloc(struct feature_node, max_nr_attr);
	dvec_t predicted_ctr;
	dvec_t real_ctr;
	ivec_t num_clicks;
	ivec_t num_impressions;

	int n;
	if (model_->bias >= 1)
		n = get_nr_feature(model_) + 1;
	else
		n = get_nr_feature(model_);

	int nr_class = get_nr_class(model_);
	double *prob_estimates = NULL;
	prob_estimates = (double *) malloc(nr_class * sizeof(double));

	labels = Malloc(int, nr_class);
	get_labels(model_, labels);

	max_line_len = 1024;
	line = (char *) malloc(max_line_len * sizeof(char));
	int clicks = 0;
	int shows = 0;
	while (readline(input) != NULL) {
		int i = 0;
		//double target_label, predict_label;
		double target_ctr, predict_ctr;
		char *idx, *val, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		/*
		 label = strtok(line," \t");
		 target_label = strtod(label,&endptr);
		 if(endptr == label)
		 exit_input_error(total+1);
		 */
		char *p = strtok(line, " \t\n"); //clicks
		if (p == NULL) // empty line
			exit_input_error(total + 1);

		clicks = atoi(p);
		p = strtok(NULL, " \t"); // shows
		shows = atoi(p);
		p = strtok(NULL, " \t"); // qid:1

		if (shows <= 0 || clicks > shows) {
			continue;
		}

		target_ctr = (double) clicks / shows;

		while (1) {
			if (i >= max_nr_attr - 2)	// need one more for index = -1
					{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,
						max_nr_attr * sizeof(struct feature_node));
			}

			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0'
					|| x[i].index <= inst_max_index)
				exit_input_error(total + 1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val, &endptr);
			if (endptr == val || errno != 0
					|| (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total + 1);

			++i;
		}

		if (model_->bias >= 0) {
			x[i].index = n;
			x[i].value = model_->bias;
			++i;
		}

		x[i].index = -1;

		predict_probability(model_, x, prob_estimates);
		fprintf(output, "%d %d", clicks, shows);
		predict_ctr = prob_estimates[0];
		fprintf(output, " %g\n", predict_ctr);

		num_clicks.push_back(clicks);
		num_impressions.push_back(shows);
		real_ctr.push_back((double) clicks / shows);
		predicted_ctr.push_back(predict_ctr);
	}

	total = real_ctr.size();
	printf("total  %d\n", total);
	double max_auc = auc_ctr(real_ctr, num_clicks, num_impressions);
	printf("maxAUC %g\n", max_auc);

	double auc = auc_ctr(predicted_ctr, num_clicks, num_impressions);
	printf("AUC    %g\n", auc);
	printf("NAUC   %g\n", auc / max_auc);

	free(labels);
	free(prob_estimates);
	free(x);
}
