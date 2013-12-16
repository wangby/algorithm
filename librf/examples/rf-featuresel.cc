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
    CmdLine cmd("featuresel", ' ', "0.1");
    SwitchArg headerFlag("","header","CSV file has a var name header",false);
    ValueArg<string> delimArg("","delim","CSV delimiter", false,",","delimiter");
    ValueArg<string>  dataArg("d", "data",
                                   "Training Data", true, "", "trainingdata");
    ValueArg<int> numfeaturesArg("f", "features", "# features", false,
                                 -1, "int");
    ValueArg<string> rankingArg("r", "rankings", "rankings", true, "", "rankings");
    ValueArg<string> outArg("o", "output", "output", true, "", "output");
    ValueArg<string> labelArg("l", "labels", "labels", true, "", "labels");

    cmd.add(outArg);
    cmd.add(labelArg);
    cmd.add(rankingArg);
    cmd.add(delimArg);
    cmd.add(headerFlag);
    cmd.add(numfeaturesArg);
    cmd.add(dataArg);

    cmd.parse(argc, argv);
    bool header = headerFlag.getValue();
    string delim = delimArg.getValue();
    string datafile = dataArg.getValue();
    string rankingfile = rankingArg.getValue();
    string outfile = outArg.getValue();
    string labelfile = labelArg.getValue();

    int num_features = numfeaturesArg.getValue();
    InstanceSet* set =  InstanceSet::load_csv_and_labels(datafile, labelfile, header, delim);
    ifstream in(rankingfile.c_str());
    vector<int> topn;
    for (int i = 0; i < num_features; ++i) {
      int feature_no;
      float feature_score;
      in >> feature_no >> feature_score;
      topn.push_back(feature_no);
    }
    InstanceSet* feature_sel = InstanceSet::feature_select(*set, topn);
    ofstream out(outfile.c_str());
    feature_sel->write_csv(out, false, " ");
    delete set;
    delete feature_sel;
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
