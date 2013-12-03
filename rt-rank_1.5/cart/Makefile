linuxgcc:
	g++ main.cpp impurity.cpp tuple.cpp boosted_classify.cpp forest.cpp -lboost_thread -lpthread -O4 -o rtrank
mac:
	g++ main.cpp impurity.cpp tuple.cpp boosted_classify.cpp forest.cpp -I/opt/local/include/ -L/opt/local/lib -lboost_thread-mt -O4 -o rtrank
intel:
	icc main.cpp impurity.cpp tuple.cpp boosted_classify.cpp forest.cpp -lboost_thread -lpthread -static -fast -o rtrank
