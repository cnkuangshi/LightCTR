export CC  = gcc
export CXX = g++
export CFLAGS = -std=c++11 -Wall -O3 -D__AVX2__ -mavx -mssse3 -Wno-unknown-pragmas -Wno-reorder -Wno-conversion-null -Wno-strict-aliasing -Wno-sign-compare

BIN = LightCTR_BIN
ZMQ_INC = ./LightCTR/third/zeromq/include
ZMQ_LIB = ./LightCTR/third/zeromq/lib/libzmq.a
OBJ =
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm -ldl

STANDALONE = *.cpp LightCTR/*.h LightCTR/common/*.h LightCTR/predict/*.h LightCTR/predict/*.cpp LightCTR/util/*.h LightCTR/train/*.h LightCTR/train/*.cpp LightCTR/train/layer/*.h LightCTR/train/unit/*.h
DISTRIBUT = $(STANDALONE) $(ZMQ_INC) LightCTR/distribut/*.h

LightCTR_BIN : $(STANDALONE)
master : $(DISTRIBUT)
ps : $(DISTRIBUT)
worker : $(DISTRIBUT)
ring_master : $(DISTRIBUT)
ring_worker : $(DISTRIBUT)

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

master :
	$(CXX) $(CFLAGS) -o LightCTR_BIN_Master $(filter %.cpp %.o %.c, $^) -D MASTER -Xlinker $(ZMQ_LIB) $(LDFLAGS)

ps :
	$(CXX) $(CFLAGS) -o LightCTR_BIN_PS $(filter %.cpp %.o %.c, $^) -D PS -Xlinker $(ZMQ_LIB) $(LDFLAGS)

worker :
	$(CXX) $(CFLAGS) -o LightCTR_BIN_Worker $(filter %.cpp %.o %.c, $^) -D WORKER -Xlinker $(ZMQ_LIB) $(LDFLAGS)

ring_master :
	$(CXX) $(CFLAGS) -o LightCTR_BIN_Ring_Master $(filter %.cpp %.o %.c, $^) -D MASTER_RING -Xlinker $(ZMQ_LIB) $(LDFLAGS)

ring_worker :
	$(CXX) $(CFLAGS) -o LightCTR_BIN_Ring_Worker $(filter %.cpp %.o %.c, $^) -DWORKER_RING -DTEST_CNN -Xlinker $(ZMQ_LIB) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN) $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
