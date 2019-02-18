//
//  poolingLayer.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/24.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef poolingLayer_h
#define poolingLayer_h

#include <vector>
#include "../../util/matrix.h"
#include "layer_abst.h"

struct Pool_Config {
    size_t size;
};
// Pooling or Maxout
// TODO K-Max Pooling
template <typename ActivationFunction>
class Max_Pooling_Layer : public Layer_Base {
public:
    Max_Pooling_Layer(Layer_Base* _prevLayer, size_t _dimention, Pool_Config _config):
    Layer_Base(_prevLayer, _dimention, _dimention), config(_config) {
        this->activeFun = new ActivationFunction();
        assert(this->input_dimention == this->output_dimention);
        
        printf("Pooling Layer\n");
    }
    Max_Pooling_Layer() = delete;
    
    ~Max_Pooling_Layer() {
    }
    
    void registerGradient(std::shared_ptr<BufferFusion<float> > _buf_fusion) {
        if (this->nextLayer) {
            this->nextLayer->registerGradient(_buf_fusion);
        }
    }
    
    vector<float>* forward(vector<Matrix*>* prevLOutput) {
        assert(prevLOutput->size() == this->input_dimention);
        
        // init ThreadLocal var
        vector<Matrix*>*& output_act = *tl_output_act;
        if (output_act == NULL) {
            output_act = new vector<Matrix*>();
            output_act->resize(this->output_dimention);
        }
        vector<Matrix*>*& input_delta = *tl_input_delta;
        if (input_delta == NULL) {
            input_delta = new vector<Matrix*>();
            input_delta->resize(this->input_dimention);
        }
        
        // do Max pooling
        FOR(feamid, this->input_dimention) {
            Matrix* mat = prevLOutput->at(feamid);
            
            assert(mat->x_len >= config.size && mat->y_len >= config.size);
            
            if (input_delta->at(feamid) == NULL) {
                output_act->at(feamid) = new Matrix((mat->x_len - config.size) / config.size + 1,
                                                    (mat->y_len - config.size) / config.size + 1);
                input_delta->at(feamid) = new Matrix(mat->x_len, mat->y_len);
            }
            
            auto cur_out = output_act->at(feamid);
            cur_out->zeroInit();
            auto cur_in = input_delta->at(feamid);
            cur_in->zeroInit();
            for (size_t i = 0; i < mat->x_len - config.size + 1; i+= config.size) {
                for (size_t j = 0; j < mat->y_len - config.size + 1; j+=config.size) {
                    float MaxV = -0x3fffffff;
                    size_t mx = -1, my = -1;
                    for (size_t x = i; x < i + config.size; x++) {
                        for (size_t y = j; y < j + config.size; y++) {
                            if (MaxV < *mat->getEle(x, y)) {
                                MaxV = *mat->getEle(x, y);
                                mx = x, my = y;
                            }
                        }
                    }
                    assert(mx != -1 && my != -1);
                    *cur_out->getEle(i / config.size, j / config.size) = MaxV;
                    *cur_in->getEle(mx, my) = 1;
                }
            }
        }
        return this->nextLayer->forward(output_act);
    }
    
    void backward(vector<Matrix*>* outputDelta) {
        assert(outputDelta->size() == this->output_dimention);
        
        vector<Matrix*>*& input_delta = *tl_input_delta;
        assert(input_delta);
        
        // Unpooling
        FOR(fid, this->input_dimention) {
            Matrix* mat = input_delta->at(fid);
            for (size_t i = 0; i < mat->x_len - config.size + 1; i+= config.size) {
                for (size_t j = 0; j < mat->y_len - config.size + 1; j+=config.size) {
                    // loop pooling size
                    for (size_t x = i; x < i + config.size; x++) {
                        for (size_t y = j; y < j + config.size; y++) {
                            if (*mat->getEle(x, y) > 0) {
                                *mat->getEle(x, y) = *outputDelta->at(fid)->getEle(i / config.size, j / config.size);
                            }
                        }
                    }
                }
            }
        }
        return this->prevLayer->backward(input_delta);
    }
    
    const vector<Matrix*>* output() {
        vector<Matrix*>*& output_act = *tl_output_act;
        assert(output_act);
        return output_act;
    }
    
private:
    Pool_Config config;
    ThreadLocal<vector<Matrix*>*> tl_output_act;
    ThreadLocal<vector<Matrix*>*> tl_input_delta; // mask of max position
};

#endif /* poolingLayer_h */
