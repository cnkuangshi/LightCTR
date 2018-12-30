export CC  = gcc
export CXX = g++
export CFLAGS = -std=c++11 -Wall -O3 -D__AVX__ -mavx -mssse3 -Wno-unknown-pragmas -Wno-reorder -Wno-null-conversion

BIN = LightCTR_BIN
ZMQ_INC = ./LightCTR/third/zeromq/include
ZMQ_LIB = ./LightCTR/third/zeromq/lib/libzmq.a
OBJ =
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm

STANDALONE = *.cpp $(ZMQ_INC) LightCTR/*.h LightCTR/common/*.h LightCTR/predict/*.h LightCTR/predict/*.cpp LightCTR/util/*.h LightCTR/train/*.h LightCTR/train/*.cpp LightCTR/train/layer/*.h LightCTR/train/unit/*.h
DISTRIBUT = $(STANDALONE) LightCTR/distribut/*.h

LightCTR_BIN : $(STANDALONE)
master : $(DISTRIBUT)
ps : $(DISTRIBUT)
worker : $(DISTRIBUT)
ring_master : $(DISTRIBUT)
ring_worker : $(DISTRIBUT)

$(BIN) :
	$(CXX) $(CFLAGS) -Xlinker $(ZMQ_LIB) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

master :
	$(CXX) $(CFLAGS) -Xlinker $(ZMQ_LIB) $(LDFLAGS) -o LightCTR_BIN_Master $(filter %.cpp %.o %.c, $^) -D MASTER

ps :
	$(CXX) $(CFLAGS) -Xlinker $(ZMQ_LIB) $(LDFLAGS) -o LightCTR_BIN_PS $(filter %.cpp %.o %.c, $^) -D PS

worker :
	$(CXX) $(CFLAGS) -Xlinker $(ZMQ_LIB) $(LDFLAGS) -o LightCTR_BIN_Worker $(filter %.cpp %.o %.c, $^) -D WORKER

ring_master :
	$(CXX) $(CFLAGS) -Xlinker $(ZMQ_LIB) $(LDFLAGS) -o LightCTR_BIN_Ring_Master $(filter %.cpp %.o %.c, $^) -D MASTER_RING

ring_worker :
	$(CXX) $(CFLAGS) -Xlinker $(ZMQ_LIB) $(LDFLAGS) -o LightCTR_BIN_Ring_Worker $(filter %.cpp %.o %.c, $^) -D WORKER_RING

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN) $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
