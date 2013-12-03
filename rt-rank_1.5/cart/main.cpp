//main.cpp

#include "main.h"

#define REG
//#define FAST

int main(int argc, char* argv[]) {
  int i;
  srand(time(NULL));


  // get command line args
  args_t myargs;
  init_args(myargs);
  if (!get_args(argc, argv, myargs)) {
    printf("RT-Rank Version 1.5 (alpha) usage: [-options] train.txt test.txt output.txt [test.txt output.txt]*\n");
	printf("\nRequired flags:\n");
	printf("-f int\tNumber of features in the data sets. \n");
	printf("You must use one of the following:\n");
	//printf(" -R \trun regression trees.\n");

	printf("\nOptional flags:\n");
	printf("-B \tgradient boosting mode (standalone).\n");
	printf("-F \trun random forest (standalone).\n");
	printf("-a float\t stepsize.\n");
	printf("-d int \tmax treep depth (for gradient boosting trees are typically limited to a small depth, e.g. d=4).\n");
	printf("-p int\tnumber of processors/threads to use.\n");
	printf("-k \tnumber of randomly selected features used for building each trees of a random forest.\n");
	printf("-t int \tnumber of trees for random forest.\n");
	printf("-m \tuse mode for prediction at leaf nodes (default is mean)\n");
	printf("-z \tsubstitute missing features with zeros (recommended if missing features exist).\n");
	printf("-e \tuse entropy to measure impurity for CART (default is squared loss).\n");

	printf("\nOperation in wrapper mode (e.g. wih Python scripts):\n");
	printf("-w \tread in weights.\n");
	printf("-r \tnumber of trees (/iterations).\n");
	printf("-s \tprint the set of features used to built the tree to stdout.\n");
	printf("\n\n");
    return 0;
  }

  // load data from input files
  data_t train;
  vec_data_t test;
  if (!load_data(train, test, myargs)) {
    printf("could not load data files\n"); 
    return 0;
  }

  // presort the training data for each feature
  add_idx(train);  
  myargs.ntra=train.size();
 if (myargs.processors==1)
	presort(train, myargs);
 else
	presort_p(train, myargs);

  for (int round = 0; round < myargs.rounds; round++) 
  {
    // allocate memory for predictions
    preds_t train_preds;
    vec_preds_t test_preds;
    init_vec(train, train_preds, test, test_preds);

    // if enabled, get data from stdin
    if (myargs.read_targets)
      read_targets(train, myargs);
    if (myargs.read_weights)
      read_weights(train, myargs);
   
    // ***do algorithm*** //


    // single regression tree
    int N = train.size();
	vector<int> countData(myargs.ntra, 1);
    if (myargs.alg == ALG_BOOST || myargs.alg == ALG_REGRESSION) {
      for (int T = 0; T < myargs.trees; T++) {
	// make tree
	dt_node* t = new dt_node(train, myargs, myargs.depth, 1, myargs.kfeatures, false, myargs);

	// get classification on training data
	double train_rmse = dt_node::classify_all(train, t, train_preds, myargs);

	// update targets
	for (i=0; i<N; i++)
	  train[i]->target = train[i]->label - train_preds[i];
	if (myargs.verbose) fprintf(stderr, "%d,%f", T, (float)train_rmse);
      
	// classify test data
	for (i=0; i<myargs.num_test; i++) {
	  double test_rmse = dt_node::classify_all(test[i], t, test_preds[i], myargs);
	  // write output every 50 trees
	  //if (T % 50 == 0)
	  //  tuple::write_to_file(test_preds[i], test[i], myargs.test_outs[i]);
	  if (myargs.verbose) fprintf(stderr, ",%f", (float)test_rmse);
	}      

	// if combining with python, print predictions of this tree to stdout
	if (myargs.read_targets)
	  for(i=0; i<N; i++)
	    cout << train_preds[i] << endl;
	if (myargs.read_targets)
	  for (i=0; i<myargs.num_test; i++)
	    for(int j = 0; j < test_preds[i].size(); j++)
	      cout << test_preds[i][j] << endl;
	if (myargs.print_features){
	  t->print_features();
	  cout<<endl;
	}
   
	// delete tree
	delete t;
	if (myargs.verbose) fprintf(stderr, "\n");
      }

      // write final predictions
      for (i=0; i<myargs.num_test; i++) {
	tuple::write_to_file(test_preds[i], test[i], myargs.test_outs[i]);
      }
    }

    // forest
    if (myargs.alg == ALG_FOREST) {
      random_forest_p(train, test, train_preds, test_preds, myargs);
      //cout << "rmse = " << rmse(test_preds[0], test[0]) << endl;
      fprintf(stderr, "rmse=%f\n", (float)(rmse(test_preds[0], test[0])));

	// if combining with python, print predictions of this tree to stdout
      if (myargs.read_targets) {
	for(i=0; i<N; i++)
	  cout << train_preds[i] << endl;
      	for (i=0; i<myargs.num_test; i++)
	  for(int j = 0; j < test_preds[i].size(); j++)
	    cout << test_preds[i][j] << endl;
      }
    }
  }

  // free memory and exit
  free_memory(myargs, train, test);
  return 0;
}
