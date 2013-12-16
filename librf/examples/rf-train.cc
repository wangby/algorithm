#include "librf/librf.h"
#include <sstream>
#include <tclap/CmdLine.h>
#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;
using namespace librf;
using namespace TCLAP;
int main(int argc, char*argv[]) {
  // Check arguments
  try {
    CmdLine cmd("rf-train", ' ', "0.1");
    SwitchArg csvFlag("","csv","Data is a CSV file",false);
    SwitchArg headerFlag("","header","CSV file has a var name header",false);
    ValueArg<string> delimArg("","delim","CSV delimiter", false,",","delimiter");
    ValueArg<string>  dataArg("d", "data",
                                   "Training Data", true, "", "trainingdata");
    ValueArg<string> modelArg("m", "model",
                              "Model file output", true, "", "rfmodel");
    ValueArg<string> labelArg("l", "label",
                              "Label file", false, "", "labels");
    ValueArg<int> numfeaturesArg("f", "features", "# features", false,
                                 -1, "int");
    ValueArg<int> treesArg("t", "trees", "# Trees", false, 10, "int");
    ValueArg<int> kArg("k", "vars", "# vars per tree", false,
                                 -1, "int");
    ValueArg<string> probArg("p", "probfile",
                              "probability file", false, "", "probfile");
    ValueArg<string> proxArg("", "proxfile",
                              "proximity file", false, "", "proxfile");
    ValueArg<string> outliersArg("", "outliers", "outlier file", false, "outliers", "outlierfile");
    ValueArg<string> importArg("","importance", "importance", false, "", "importance");
    SwitchArg unsuperFlag("", "unsupervised", "Unsupervised mode", false);

    cmd.add(outliersArg);
    cmd.add(unsuperFlag);
    cmd.add(delimArg);
    cmd.add(importArg);
    cmd.add(headerFlag);
    cmd.add(csvFlag);
    cmd.add(labelArg);
    cmd.add(numfeaturesArg);
    cmd.add(dataArg);
    cmd.add(modelArg);
    cmd.add(treesArg);
    cmd.add(kArg);
    cmd.add(probArg);
    cmd.add(proxArg);
    cmd.parse(argc, argv);

    bool csv = csvFlag.getValue();
    bool header = headerFlag.getValue();
    bool unsupervised = unsuperFlag.getValue();
    string outlier_file = outliersArg.getValue();
    string delim = delimArg.getValue();
    string datafile = dataArg.getValue();
    string modelfile = modelArg.getValue();
    string labelfile = labelArg.getValue();
    string probfile = probArg.getValue();
    string proxfile = proxArg.getValue();
    string importfile = importArg.getValue();
    int K = kArg.getValue();
    int num_features = numfeaturesArg.getValue();
    int num_trees = treesArg.getValue();
    InstanceSet* set = NULL;
    unsigned int seed = 1;
    int set_size;
    //if (!csv) {
      // set = InstanceSet::load_libsvm(datafile, num_features);
    //} else {
    if (!unsupervised) {
      set = InstanceSet::load_csv_and_labels(datafile, labelfile, header, delim);
      set_size = set->size();
    } else {
      set = InstanceSet::load_unsupervised(datafile, &seed, header, delim);
      set_size = set->size() / 2;
    }
    //}
    // if mtry was not set defaults to sqrt(num_features)
    if (K == -1) {
       K = int(sqrt(double(set->num_attributes())));
    }
    // vector<int> weights;
    RandomForest rf(*set, num_trees, K); //, weights);
    cout << "Training Accuracy " << rf.training_accuracy() << endl;
    cout << "OOB Accuracy " << rf.oob_accuracy() << endl;
    cout << "---Confusion Matrix----" << endl;
    rf.oob_confusion();
    vector<pair<float, float> > rd;
    vector<int> hist;
    cout << "Reliability Diagram" << endl;
    rf.reliability_diagram(10, &rd, &hist, 0);
    cout << "bin fraction 1 0 total" << endl;
    for (int i = 0; i < rd.size(); ++i) {
      int positive = int(round(hist[i]*rd[i].second));
      cout << rd[i].first << " " << rd[i].second << " ";
      cout << positive << " " << (hist[i] - positive) << " " << hist[i] <<endl;
    }

    ofstream out(modelfile.c_str());
    rf.write(out);
    cout << "Model file saved to " << modelfile << endl;

    if (probfile.size() > 0) {
      ofstream prob_out(probfile.c_str());
      for (int i = 0; i < set->size(); i++) {
        prob_out << rf.oob_predict_prob(i, 0) << endl;
      }
    }
    if (proxfile.size() > 0) {
      cout << "Generating proximity matrix" << endl;
      ofstream prox_out(proxfile.c_str());
      vector<vector<float> > mat(set_size, vector<float>(set_size, 0.0));
      rf.compute_proximity(*set, &mat, set_size);
      for (int i = 0; i < mat.size(); ++i) {
        for (int j = 0; j < mat[i].size(); ++j) {
          prox_out << mat[i][j] << " ";
        }
        prox_out << endl;
      }
      ofstream out_file(outlier_file.c_str());
      vector<pair<float, int> >outliers;
      rf.compute_outliers(*set, 0, mat, &outliers);
      for (int i = 0; i < outliers.size(); ++i) {
        out_file << outliers[i].second << " " << outliers[i].first << endl;
      }
    }

    if (importfile.size() > 0) {
      ofstream rankings(importfile.c_str());
      vector< pair<float, int> > scores;
      rf.variable_importance(&scores, &seed);
      for (int i = 0; i < scores.size(); ++i) {
        rankings << scores[i].second << " " << scores[i].first << endl;
      }
    }
    // rf.print();
    delete set;
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
