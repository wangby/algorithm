#include "weights.h"
#include <UnitTest++.h>
#include <iostream>
using namespace std;

TEST(WeightTest) {
  weight_list w(8,4);
  w.add(4);
  w.add(5);
  w.add(6);
  w.add(7);
  for (int i = 0; i < 8; ++i) {
    cout << i << ":" << int(w[i]) << endl;
  }

}
