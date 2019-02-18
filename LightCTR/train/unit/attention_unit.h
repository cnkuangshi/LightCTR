//
//  attention_unit.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/2.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef attention_unit_h
#define attention_unit_h

#include <vector>
#include "../../util/matrix.h"
#include "../layer/fullyconnLayer.h"

// Attention-based Encoder-Decoder build a RNN that has alignment attention
template <typename ActivationFunction>
class Attention_Unit : public Layer_Base {
public:
    Attention_Unit(size_t _dimention, size_t _hidden_size, size_t _recurrent_cnt):
    Layer_Base(NULL, _recurrent_cnt, _dimention), dimention(_dimention), batch_size(_recurrent_cnt) {
        this->activeFun = new ActivationFunction();
        
        printf("Attention-based Unit\n");
        // alpha transform is computed by DxH and Hx1 fc Layer
        printf("-- Attention Inner FC-1 ");
        transformFunc = new Fully_Conn_Layer<Sigmoid>(NULL, _dimention, _hidden_size);
        transformFunc->needInputDelta = true;
        printf("-- Attention Inner FC-2 ");
        transformFunc_bp = new Fully_Conn_Layer<Sigmoid>(transformFunc, _hidden_size, 1);
    }
    Attention_Unit() = delete;
    
    ~Attention_Unit() {
        delete transformFunc_bp;
        delete transformFunc;
    }
    
    // Attention input data should be data concating rnn encoder output sequence, rather than one cell's output
    vector<float>* forward(vector<Matrix*>* const prevLOutputMatrix) {
        assert(prevLOutputMatrix->size() == batch_size);
        
        // init threadlocal var
        vector<Matrix*>*& input = *tl_input;
        if (input == NULL) {
            input = new vector<Matrix*>();
            input->resize(batch_size);
        }
        Matrix*& attentionOutput = *tl_attentionOutput;
        if (attentionOutput == NULL) {
            attentionOutput = new Matrix(1, dimention);
        }
        Matrix*& fc_output_act = *tl_fc_output_act;
        if (fc_output_act == NULL) {
            fc_output_act = new Matrix(1, batch_size);
        }
        Matrix*& cache = *tl_cache;
        
        vector<Matrix*>*& wrapper = *tl_wrapper;
        if (wrapper == NULL) {
            wrapper = new vector<Matrix*>();
            wrapper->resize(1);
        }
        
        FOR(idx, prevLOutputMatrix->size()) {
            input->at(idx) = prevLOutputMatrix->at(idx)->copy(input->at(idx)); // 1xD
            assert(input->at(idx)->size() == dimention);
            wrapper->at(0) = input->at(idx);
            auto res = transformFunc->forward(wrapper);
            assert(res->size() == 1);
            *fc_output_act->getEle(0, idx) = res->at(0);
        }
        // Softmax normalization
        softmax.forward(fc_output_act->pointer());
        
        attentionOutput->zeroInit();
        FOR(idx, prevLOutputMatrix->size()) {
            cache = input->at(idx)->copy(cache)->scale(*fc_output_act->getEle(0, idx));
            attentionOutput->add(cache);
        }
        
        return attentionOutput->pointer();
    }
    
    void backward(vector<Matrix*>* const outputDeltaMatrix) {
        Matrix* outputDelta = outputDeltaMatrix->at(0);
        assert(outputDelta->size() == this->output_dimention);
        
        // init threadlocal var
        vector<Matrix*>*& input = *tl_input;
        assert(input);
        Matrix*& fc_output_act = *tl_fc_output_act;
        assert(fc_output_act);
        vector<Matrix*>*& wrapper = *tl_wrapper;
        assert(wrapper);
        vector<float>*& scaleDelta = *tl_scaleDelta;
        if (scaleDelta == NULL) {
            scaleDelta = new vector<float>();
            scaleDelta->resize(batch_size);
        }
        vector<Matrix*>*& input_delta = *tl_input_delta;
        if (input_delta == NULL) {
            input_delta = new vector<Matrix*>();
            input_delta->resize(batch_size);
        }
        Matrix*& cache_bp = *tl_cache_bp;
        if (cache_bp == NULL) {
            cache_bp = new Matrix(1, 1);
        }
        Matrix*& cache = *tl_cache;
        assert(cache);
        
        FOR(idx, input->size()) {
            // update softmax_fc by delta of softmax_fc(X)
            auto res = input->at(idx)->Multiply(cache_bp, outputDelta->transpose());
            outputDelta->transpose(); // recover
            assert(res->size() == 1);
            scaleDelta->at(idx) = *cache_bp->getEle(0, 0);
        }
        softmax.backward(scaleDelta, fc_output_act->pointer(), scaleDelta);
        // update transformFunc
        FOR(idx, input->size()) {
            *cache_bp->getEle(0, 0) = scaleDelta->at(idx);
            wrapper->at(0) = cache_bp;
            transformFunc_bp->backward(wrapper);
            // input delta of transformFunc
            const Matrix* delta = transformFunc->inputDelta();
            input_delta->at(idx) = delta->copy(input_delta->at(idx));
        }
        // pass back delta of X
        FOR(idx, input->size()) {
            cache = outputDelta->copy(cache)->scale(*fc_output_act->getEle(0, idx));
            input_delta->at(idx)->add(cache);
        }
    }
    
    const vector<Matrix*>* output() {
        Matrix*& attentionOutput = *tl_attentionOutput;
        assert(attentionOutput);
        vector<Matrix*>*& wrapper = *tl_wrapper;
        wrapper->at(0) = attentionOutput;
        return wrapper;
    }
    vector<Matrix*>* inputDelta() {
        vector<Matrix*>*& input_delta = *tl_input_delta;
        assert(input_delta); // check layer connect
        return input_delta;
    }
    
    void applyBatchGradient() {
        transformFunc->applyBatchGradient();
        if (nextLayer) {
            nextLayer->applyBatchGradient();
        }
    }
    
private:
    Fully_Conn_Layer<Sigmoid> *transformFunc, *transformFunc_bp;
    Softmax softmax;
    size_t batch_size, dimention;
    
    ThreadLocal<vector<Matrix*>*> tl_input;
    ThreadLocal<Matrix*> tl_attentionOutput;
    ThreadLocal<Matrix*> tl_fc_output_act;
    ThreadLocal<Matrix*> tl_cache, tl_cache_bp;
    
    ThreadLocal<vector<float>*> tl_scaleDelta;
    ThreadLocal<vector<Matrix*>*> tl_input_delta;
    
    ThreadLocal<vector<Matrix*>*> tl_wrapper;
};

#endif /* attention_unit_h */
