VERSION = $(VERSION32_SIGN)

LIB = -lpthread

CC = g++

CPPFLAGS = -g -Wall -D_REENTRANT -O3 -fPIC
DEBUG_CPPFLAGS = -g -Wall -fPIC -D_REENTRANT -DDEBUG_ON -DFILE_DUMP_ON

all : libgbdt.a gbdt_train gbdt_predict libgbdt.so output

libgbdt.a : gradient_boosting.o
	ar rcv $@ $^

gbdt_train : gbdt_train.o libgbdt.a
	$(CC) -o $@ $^ $(INCLUDE) $(LIB) $(DEBUG_CPPFLAGS)

gbdt_predict : gbdt_predict.o libgbdt.a
	$(CC) -o $@ $^ $(INCLUDE) $(LIB) $(DEBUG_CPPFLAGS)

libgbdt.so : libgbdt.a
	$(CC) -shared -o $@ $^ $(LIB)  $(DEBUG_CPPFLAGS)

%.o : %.cpp
	$(CC) -c -o $@ $^ $(INCLUDE) $(DEBUG_CPPFLAGS) $(INCLUDE)

OUTPUT_DIR = ./output
LIB_DIR = $(OUTPUT_DIR)/lib
INCLUDE_DIR = $(OUTPUT_DIR)/include
BIN_DIR = $(OUTPUT_DIR)/bin

clean : 
	find . -name "*.o" -maxdepth 3 -exec rm {} \;
	find . -name "*.a" -maxdepth 3 -exec rm {} \;
	find . -name "*.so" -maxdepth 3 -exec rm {} \;
	find . -name "gbdt_train" -maxdepth 3 -exec rm {} \;
	find . -name "gbdt_predict" -maxdepth 3 -exec rm {} \;
	if [ -d $(OUTPUT_DIR) ] ; then rm -r $(OUTPUT_DIR); fi

output : libgbdt.a gradient_boosting.h gbdt_train gbdt_predict libgbdt.so
	if [ ! -d $(OUTPUT_DIR) ] ; then mkdir $(OUTPUT_DIR) ; fi
	if [ ! -d $(LIB_DIR) ] ; then mkdir $(LIB_DIR) ; fi
	if [ ! -d $(INCLUDE_DIR) ] ; then mkdir $(INCLUDE_DIR) ; fi
	if [ ! -d $(BIN_DIR) ] ; then mkdir $(BIN_DIR) ; fi

	cp gbdt_train $(BIN_DIR)
	cp gbdt_predict $(BIN_DIR)
	cp libgbdt.so $(BIN_DIR)
	cp libgbdt.a $(LIB_DIR)
	cp gradient_boosting.h $(INCLUDE_DIR)
