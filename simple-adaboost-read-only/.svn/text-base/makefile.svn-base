CC=g++
CFLAGS=-c -Wall -g
#LDFLAGS=-L /usr/local/lib 
SOURCES_TEST=ml_data.cpp base_learner.cpp base_learner_test.cpp
SOURCES=ml_data.cpp base_learner.cpp adaboost.cpp main.cpp
OBJECTS_TEST=$(SOURCES_TEST:.cpp=.o)
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE_TEST=base_learner_test
EXECUTABLE=adaboost

all: $(EXECUTABLE)

$(EXECUTABLE_TEST): $(OBJECTS_TEST)
	$(CC) $(OBJECTS_TEST) -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

%.o:%.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -fr *.o *~ $(EXECUTABLE_TEST)
