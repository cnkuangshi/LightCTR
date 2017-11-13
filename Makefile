export CC  = gcc
export CXX = g++
export CFLAGS = -std=c++11 -Wall -O3 -msse2  -Wno-unknown-pragmas -Wno-reorder -Wno-null-conversion

BIN = LightCTR_BIN
OBJ = 
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm 

LightCTR_BIN : *.cpp LightCTR/*.h LightCTR/common/*.h LightCTR/predict/*.h LightCTR/predict/*.cpp LightCTR/util/*.h LightCTR/train/*.h LightCTR/train/*.cpp LightCTR/train/layer/*.h LightCTR/train/unit/*.h

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
