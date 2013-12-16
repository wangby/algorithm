bin_PROGRAMS =  unittests
unittests_SOURCES = unittests.cc random_forest_unittest.cc instance_set_unittest.cc
INCLUDES = -I ../UnitTestPlusPlus/includes -I ../librf
LIBS = -L../UnitTestPlusPlus -L../librf -lrf -lUnitTest++
check_PROGRAMS = unittests
TESTS = unittests
CXXFLAGS = -ggdb
