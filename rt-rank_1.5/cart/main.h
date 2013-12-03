//main.h

#include "args.h"
#include "regression_tree.h"
#include "tuple.h"
#include "forest.h"
#include "epoch.h"

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>
using namespace boost;

vector<vector<int> > fInds;

// given a set of files, read them into train and test vectors
int load_data(vector<tuple*>& train, vector< vector<tuple*> >& test, const args_t& myargs) {
  int numfeatures = myargs.features;
  int missing = myargs.missing;
  char* missing_file = myargs.missing_file;

  //zeros->missing=0, missing=1, ones=-1
  int m=myargs.ones ? -1 : myargs.missing;

  fprintf(stderr, "loading training data...");
  if (!tuple::read_input(train, myargs.train_file, numfeatures, 1, m, missing_file))
    return 0;
  fprintf(stderr, "done\n");
  fprintf(stderr, "loading test data...");
  for (int i=0; i<myargs.num_test; i++) {
    vector<tuple*> t;
    if (!tuple::read_input(t, myargs.test_files[i], numfeatures, 1, m, missing_file))
      return 0;
    test.push_back(t);
  }
  fprintf(stderr, "done\n");
  return 1;
}

// read targets from stdin, assign them to the respective tuple
void read_targets(vector<tuple*>& train, const args_t& myargs) {
  int N = train.size(), i;
  int r = myargs.read_targets;
  for (i=0; i < N; i++) {
    double t;
    cin >> t;
    train[i]->set_target(t);
    train[i]->label = t;
  }
}

void read_weights(vector<tuple*>& train, const args_t& myargs) {
  int i, N=train.size();
  for (i=0; i<N; i++) {
    double w; cin >> w;
    train[i]->weight = w;
  }
}


// free all the memory we used up
void free_memory(const args_t& myargs, vector<tuple*>& train, vector< vector<tuple*> >& test) {
  int i;
  tuple::delete_data(train);
  for (i=0; i < myargs.num_test; i++)
    tuple::delete_data(test[i]);
}


// given preds and labels, get rmse
double rmse(const vector<double>& preds, const vector<tuple*>& data) {
  double r = 0;
  int i, N = data.size();
  for (i=0; i<N; i++)
    r += squared(data[i]->label - preds[i]); 
  return sqrt(1.0 / N * r);
}

// init vectors to 0
void init_vec(const vector<tuple*>& train, vector<double>& train_preds, const vector< vector<tuple*> >& test, vector< vector<double> >& test_preds) {
  int i;
  for (i = 0; i < test.size(); i++) {
    vector<double> p;
    for (int j = 0; j < test[i].size(); j++)
      p.push_back(0.0);
    test_preds.push_back(p);
  }
  for (i = 0; i < train.size(); i++)
    train_preds.push_back(0.0);
}

/*void init_pred_vec(const vec_data_t& test, vec_preds_t& preds) {
  for (int t = 0; t < test.size(); t++) {
    preds_t p;
    for (int i = 0; i < test[t].size(); i++)
      p.push_back(0);
    preds.push_back(p);
  }
  }*/

void add_idx(vector<tuple*>& train) {
        int i, N = train.size();
        for (i = 0; i < N; i ++){
                train[i]->idx=i;
        }
}


bool mysortpred2(const pair<tuple*, int> tk1, const pair<tuple*, int > tk2) {
  return tk1.first->features[tk1.second] < tk2.first->features[tk2.second];
}

void presort(data_t train, const args_t& myargs){
	int numfeatures = myargs.features;
        fInds.push_back(vector<int>());
        // sort data
        for (int f = 1; f < numfeatures; f ++){
		fInds.push_back(vector<int>());
                vector< pair<tuple*, int > > tk;
                int z;
                for (z = 0; z < train.size(); z++)
                        tk.push_back( pair<tuple*, int >(train[z], f) );
                sort(tk.begin(), tk.end(), mysortpred2);
                for (z = 0; z < tk.size(); z++)
			fInds[f].push_back(tk[z].first->idx);
                        //data[z] = tk[z].first;
        }
}

void presort_in_range(data_t train, const args_t& myargs, int start, int end){
        // sort data
        for (int i = start; i < end; i ++){
		int f = i+1;
                vector< pair<tuple*, int > > tk;
                int z;
                for (z = 0; z < train.size(); z++)
                        tk.push_back( pair<tuple*, int >(train[z], f) );
                sort(tk.begin(), tk.end(), mysortpred2);
                for (z = 0; z < tk.size(); z++)
                        fInds[f][z]=tk[z].first->idx;
                        //data[z] = tk[z].first;
        }
}

void presort_p(data_t train, const args_t& myargs) {
	int numfeatures = myargs.features;
	int n = train.size();
	for (int f = 0; f < numfeatures; f ++)
		fInds.push_back(vector<int>(n, 0));
	
	int numthreads = myargs.processors;
	thread** threads = new thread*[numthreads];	
	for (int i = 0; i < numthreads; i ++)
		threads[i] = new thread(bind(presort_in_range, train, cref(myargs), i*(numfeatures-1)/numthreads, (i+1)*(numfeatures-1)/numthreads));
  
	for (int i = 0; i < numthreads; i++) {
    		threads[i]->join();
    		delete threads[i];
  	}
  	delete[] threads;
}

