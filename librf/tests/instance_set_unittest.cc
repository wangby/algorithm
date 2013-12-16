#include "librf/instance_set.h"
#include <UnitTest++.h>
#include <iostream>
using namespace std;
using namespace librf;
struct InstanceSetFixture {
	// Do some Setup
	InstanceSetFixture() {
    // Load example libSVM file
    csv = InstanceSet::load_csv_and_labels("../data/heart.csv",
                                           "../data/heart_labels.txt",true);
  }
	// Do some Teardown
	~InstanceSetFixture() {
    delete csv;
	}
  InstanceSet* csv;
};

TEST_FIXTURE(InstanceSetFixture, SanityCheck)
{
  CHECK_EQUAL(csv->size(), 270);
}


/*
int main()
{
    return UnitTest::RunAllTests();
}
*/

