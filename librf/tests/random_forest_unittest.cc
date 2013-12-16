#include "librf/random_forest.h"
#include "librf/instance_set.h"
#include <UnitTest++.h>
#include <iostream>
#include <fstream>
using namespace std;
using namespace librf;


struct RF_TrainPredictFixture {
  RF_TrainPredictFixture() {
    cout << "loading heart data" << endl;
//    heart_ = InstanceSet::load_libsvm("../data/heart.svm", 14);
    heart_ = InstanceSet::load_csv_and_labels("../data/heart.csv",
                                           "../data/heart_labels.txt",true);
  }
  ~RF_TrainPredictFixture() {
    delete heart_;
  }
  InstanceSet* heart_;
};

TEST_FIXTURE(RF_TrainPredictFixture, TrainPredictCheck) {

  RandomForest rf(*heart_, 100, 12);
  // rf.print();
  cout << "Training accuracy " << rf.training_accuracy() <<endl;
  cout << "OOB Accuracy " << rf.oob_accuracy() <<endl;
  rf.oob_confusion();
  cout << "Test accuracy " << rf.testing_accuracy(*heart_) <<endl;
  ofstream out("out.test");
  cout <<"save test" << endl;
  rf.write(out);
  ifstream in ("out.test");
  RandomForest loaded;
  loaded.read(in);
  unsigned int seed = 1;
  vector< pair<float, int> > scores;
  rf.variable_importance(&scores, &seed);
  for (int i = 0; i < scores.size(); ++i) {
    cout << heart_->get_varname(scores[i].second) << ":" << scores[i].first <<endl;
  }
}
/*
int main()
{
    return UnitTest::RunAllTests();
}*/

