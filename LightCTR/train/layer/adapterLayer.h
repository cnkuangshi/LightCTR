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
    Layer_Base(_prevLayer, _prevLayer->output_dimension, _prevLayer->output_dimension) {
        this->activeFun = new ActivationFunction();
        this->output_dimension *= flatten_cnt * flatten_cnt;
        
        printf("Adapter Layer\n");
    }
    Adapter_Layer() = delete;
    
    ~Adapter_Layer() {
    }
    
    vector<float>& forward(const vector<Matrix*>& prevLOutput) {
        // init ThreadLocal var
        Matrix& output_act = *tl_output_act;
        MatrixArr& input_delta = *tl_input_delta;
        // indicate lazy init once
        assert(this->output_dimension == prevLOutput.size() * prevLOutput[0]->size());
        output_act.reset(1, this->output_dimension);
        input_delta.arr.resize(this->input_dimension);
        FOR(i, this->input_dimension) {
            if (!input_delta.arr[i]) {
                input_delta.arr[i] =
                    new Matrix(prevLOutput[0]->x_len, prevLOutput[0]->y_len);
            }
        }
        
        FOR(i, prevLOutput.size()) {
            size_t offset = i * prevLOutput[i]->x_len * prevLOutput[i]->y_len;
            // Flatten data row
            FOR(x, prevLOutput[i]->x_len) {
                FOR(y, prevLOutput[i]->y_len) {
                    *output_act.getEle(0, offset + x * prevLOutput[i]->y_len + y) =
                            *prevLOutput[i]->getEle(x, y);
                }
            }
        }
        
        // init threadlocal wrapper
        vector<Matrix*>& wrapper = *tl_wrapper;
        wrapper.resize(1);
        wrapper[0] = &output_act;
        return this->nextLayer->forward(wrapper);
    }
    
    void backward(const vector<Matrix*>& outputDeltaMatrix) {
        vector<float>* outputDelta = outputDeltaMatrix[0]->pointer();
        assert(outputDelta->size() == this->output_dimension);
        
        MatrixArr& input_delta = *tl_input_delta;
        
        FOR(i, this->input_dimension) {
            auto m_ptr = input_delta.arr[i];
            assert(m_ptr);
            size_t offset = i * m_ptr->size();
            FOR(x, m_ptr->x_len) {
                FOR(y, m_ptr->y_len) {
                    *m_ptr->getEle(x, y) = outputDelta->at(offset + x * m_ptr->y_len + y);
                }
            }
        }
        this->prevLayer->backward(input_delta.arr);
    }
    
    const vector<Matrix*>& output() {
        Matrix& output_act = *tl_output_act;
        vector<Matrix*>& wrapper = *tl_wrapper;
        wrapper[0] = &output_act;
        return wrapper;
    }
    
private:
    ThreadLocal<Matrix> tl_output_act; // wx + b with activation
    ThreadLocal<MatrixArr> tl_input_delta; // delta of prevLayer wx+b Z_(L-1)
    
    ThreadLocal<vector<Matrix*> > tl_wrapper;
};

#endif /* adapterLayer_h */
