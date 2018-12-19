//
//  layer_abst.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/20.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef layer_abst_h
#define layer_abst_h

#include <cstdio>
#include <mutex>
#include "../../common/thread_pool.h"
#include "../../common/lock.h"
#include "../../util/activations.h"
#include "../../util/matrix.h"
#include "../../util/gradientUpdater.h"
#include "../../util/momentumUpdater.h"
#include "assert.h"

#define FOR(i,n) for(size_t i = 0;i < n;i++)

class Layer_Base {
public:
    Layer_Base(Layer_Base* _prevLayer, size_t _input_dimention, size_t _output_dimention):
    input_dimention(_input_dimention), output_dimention(_output_dimention) {
        nextLayer = prevLayer = NULL;
        if (_prevLayer != NULL) {
            assert(_prevLayer->output_dimention == this->input_dimention);
            this->prevLayer = _prevLayer;
            _prevLayer->nextLayer = this;
            bInputLayer = false;
            printf("Init %zux%zu ", _input_dimention, _output_dimention);
        } else {
            bInputLayer = true;
            printf("Init Input %zux%zu ", _input_dimention, _output_dimention);
        }
    }
    Layer_Base() = delete;
    virtual ~Layer_Base() {
    }
    
    virtual vector<double>* forward(vector<Matrix*>* const prevLOutputMatrix) = 0;
    
    virtual void backward(vector<Matrix*>* const outputDeltaMatrix) = 0;
    
    virtual const vector<Matrix*>* output() = 0;
    
    virtual void applyBatchGradient() { // for each mini-batch gradient batch update stage
        if (nextLayer) {
            nextLayer->applyBatchGradient();
        }
    }
    
    Activation& getActiveFun() const {
        assert(activeFun); // Notice to init activeFun in instance
        return *activeFun;
    }
    
    Activation* activeFun;
    
    Layer_Base *nextLayer, *prevLayer;
    
    size_t input_dimention, output_dimention;
    
    bool bInputLayer;
    
    SpinLock lock;
};

#endif /* layer_abst_h */
