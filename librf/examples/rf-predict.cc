#include "librf/librf.h"
#include <sstream>
#include <tclap/CmdLine.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace librf;
using namespace TCLAP;
int main(int argc, char*argv[]) {
  // Check arguments
  try {
    SwitchArg headerFlag("","header","CSV file has a var name header",false);
    ValueArg<string> delimArg("","delim","CSV delimiter", false,",","delimiter");
    ValueArg<string> labelArg("l", "label",
                              "Label file", false, "", "labels");

    CmdLine cmd("rf-predict", ' ', "0.1");
    ValueArg<string>  dataArg("d", "data",
                                   "Training Data", true, "", "testdata");
    ValueArg<string> modelArg("m", "model",
                              "Model file output", true, "", "rfmodel");

    ValueArg<int> numfeaturesArg("f", "features", "# features", false,
                                 -1, "int");
    ValueArg<string> outputArg("o", "output", "predictions", true, "", "output");
    cmd.add(delimArg);
    cmd.add(headerFlag);
    cmd.add(labelArg);
    cmd.add(outputArg);
    cmd.add(numfeaturesArg);
    cmd.add(dataArg);
    cmd.add(modelArg);
    cmd.parse(argc, argv);
    bool header = headerFlag.getValue();
    string delim = delimArg.getValue();
    string labelfile = labelArg.getValue();
    string datafile = dataArg.getValue();
    string modelfile = modelArg.getValue();
    string outfile = outputArg.getValue();

    int num_features = numfeaturesArg.getValue();
    InstanceSet* set = NULL;
    set = InstanceSet::load_csv_and_labels(datafile, labelfile, header, delim);

    RandomForest rf;
    ifstream in(modelfile.c_str());
    rf.read(in);
    cout << "Test accuracy: " << rf.testing_accuracy(*set) << endl;;
    ofstream out(outfile.c_str());
    for (int i = 0; i < set->size(); ++i) {
      out << rf.predict_prob(*set, i, 0) << endl;
    }
    cout << "Confusion matrix" << endl;
    rf.test_confusion(*set);
    vector<pair<float, float> > rd;
    vector<int> hist;
    cout << "Reliability" << endl;
    rf.reliability_diagram(*set, 10, &rd, &hist, 0);
    cout << "bin fraction 1 0 total" << endl;
    for (int i = 0; i < rd.size(); ++i) {
      int positive = int(round(hist[i]*rd[i].second));
      cout << rd[i].first << " " << rd[i].second << " ";
      cout << positive << " " << (hist[i] - positive) << " " << hist[i] <<endl;
    }

    delete set;
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
