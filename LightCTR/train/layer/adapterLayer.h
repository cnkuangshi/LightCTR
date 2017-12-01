//
//  adapterLayer.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/25.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef adapterLayer_h
#define adapterLayer_h

#include "layer_abst.h"

// Flatten and concat Matrixs into dataRow adapting CNN to FC or LSTM sequences to Attention input
template <typename ActivationFunction>
class Adapter_Layer : public Layer_Base {
public:
    Adapter_Layer(Layer_Base* _prevLayer, size_t flatten_cnt):
    Layer_Base(_prevLayer, _prevLayer->output_dimention, _prevLayer->output_dimention) {
        this->activeFun = new ActivationFunction();
        this->output_dimention *= flatten_cnt * flatten_cnt;
        
        printf("Adapter Layer\n");
    }
    ~Adapter_Layer() {
    }
    
    vector<double>* forward(vector<Matrix*>* prevLOutput) {
        // init ThreadLocal var
        Matrix*& output_act = *tl_output_act;
        vector<Matrix*>*& input_delta = *tl_input_delta;
        if (output_act == NULL) { // indicate lazy init once
            assert(this->output_dimention == prevLOutput->size() * prevLOutput->at(0)->size());
            output_act = new Matrix(1, this->output_dimention);
            input_delta = new vector<Matrix*>();
            input_delta->resize(this->input_dimention);
            FOR(i, this->input_dimention) {
                input_delta->at(i) =
                        new Matrix(prevLOutput->at(0)->x_len, prevLOutput->at(0)->y_len);
            }
        }
        
        FOR(i, prevLOutput->size()) {
            size_t offset = i * prevLOutput->at(i)->x_len * prevLOutput->at(i)->y_len;
            // Flatten data row
            FOR(x, prevLOutput->at(i)->x_len) {
                FOR(y, prevLOutput->at(i)->y_len) {
                    *output_act->getEle(0, offset + x * prevLOutput->at(i)->y_len + y) =
                            *prevLOutput->at(i)->getEle(x, y);
                }
            }
        }
        
        // init threadlocal wrapper
        vector<Matrix*>*& wrapper = *tl_wrapper;
        if (wrapper == NULL) {
            wrapper = new vector<Matrix*>();
            wrapper->resize(1);
        }
        wrapper->at(0) = output_act;
        return this->nextLayer->forward(wrapper);
    }
    
    void backward(vector<Matrix*>* outputDeltaMatrix) {
        vector<double>* outputDelta = outputDeltaMatrix->at(0)->pointer();
        assert(outputDelta->size() == this->output_dimention);
        
        vector<Matrix*>*& input_delta = *tl_input_delta;
        assert(input_delta);
        
        FOR(i, this->input_dimention) {
            auto m_ptr = input_delta->at(i);
            assert(m_ptr);
            size_t offset = i * m_ptr->size();
            FOR(x, m_ptr->x_len) {
                FOR(y, m_ptr->y_len) {
                    *m_ptr->getEle(x, y) = outputDelta->at(offset + x * m_ptr->y_len + y);
                }
            }
        }
        this->prevLayer->backward(input_delta);
    }
    
    const vector<Matrix*>* output() {
        Matrix*& output_act = *tl_output_act;
        assert(output_act);
        vector<Matrix*>*& wrapper = *tl_wrapper;
        assert(wrapper);
        wrapper->at(0) = output_act;
        return wrapper;
    }
    
private:
    ThreadLocal<Matrix*> tl_output_act; // wx + b with activation
    ThreadLocal<vector<Matrix*>*> tl_input_delta; // delta of prevLayer wx+b Z_(L-1)
    
    ThreadLocal<vector<Matrix*>*> tl_wrapper;
};

#endif /* adapterLayer_h */
