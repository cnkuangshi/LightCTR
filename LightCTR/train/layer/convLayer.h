//
//  convLayer.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/24.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef convLayer_h
#define convLayer_h

#include <vector>
#include "../../util/matrix.h"
#include "layer_abst.h"

#define FOR(i,n) for(size_t i = 0;i < n;i++)

static const bool cnn_dropout_mask[] = { // 6 * 16 sparse link matrix
    true,false,false,false,true,true,true,false,false,true,true,true,true,false,true,true,
    true,true,false,false,false,true,true,true,false,false,true,true,true,true,false,true,
    true,true,true,false,false,false,true,true,true,false,false,true,false,true,true,true,
    false,true,true,true,false,false,true,true,true,true,false,false,true,false,true,true,
    false,false,true,true,true,false,false,true,true,true,true,false,true,true,false,true,
    false,false,false,true,true,true,false,false,true,true,true,true,false,true,true,true
};

struct CNN_Config {
    size_t filter_size;
    size_t padding, stride;
};

template <typename ActivationFunction>
class Conv_Layer : public Layer_Base {
public:
    Conv_Layer(Layer_Base* _prevLayer, size_t _input_dimention,
               size_t _output_dimention, CNN_Config _config):
    Layer_Base(_prevLayer, _input_dimention, _output_dimention), config(_config) {
        init();
        printf("Convolution Layer\n");
    }
    ~Conv_Layer() {
        filterArr.clear();
        bias.clear();
        filterDelta.clear();
        biasDelta.clear();
    }
    
    void init() {
        this->activeFun = new ActivationFunction();
        // allocate filter and bias's solver memory
        updater.learnable_params_cnt(this->output_dimention * 2);
        
        filter_cnt = this->output_dimention;
        
        filterArr.resize(filter_cnt);
        FOR(i, filter_cnt) {
            filterArr[i] = new Matrix(config.filter_size, config.filter_size);
            filterArr[i]->randomInit();
        }
        
        bias.resize(filter_cnt);
        FOR(i, this->output_dimention) { // lazy init because they depend on conv result size
            bias[i] = NULL;
        }
        // init for mini-batch
        filterDelta.resize(filter_cnt);
        FOR(i, filter_cnt) {
            filterDelta[i] = new Matrix(config.filter_size, config.filter_size);
            filterDelta[i]->zeroInit();
        }
        biasDelta.resize(filter_cnt);
        FOR(i, this->output_dimention) {
            biasDelta[i] = NULL;
        }
    }
    
    vector<double>* forward(vector<Matrix*>* prevLOutput) {
        assert(this->nextLayer);
        assert(prevLOutput->size() == this->input_dimention);
        
        // init ThreadLocal var
        vector<Matrix*>*& output_act = *tl_output_act;
        if (output_act == NULL) {
            output_act = new vector<Matrix*>();
            output_act->resize(this->output_dimention);
        }
        Matrix*& cache = *tl_cache;
        
        if (this->bInputLayer) { // storage input only for input layer
            vector<Matrix*>*& input = *tl_input;
            if (input == NULL) {
                input = new vector<Matrix*>();
                input->resize(this->input_dimention);
            }
            FOR(i, this->input_dimention) {
                input->at(i) = prevLOutput->at(i);
            }
        }
        
        FOR(filid, filter_cnt) {
            auto m_ptr = output_act->at(filid);
            if (m_ptr) {
                m_ptr->zeroInit();
            }
            FOR(feamid, this->input_dimention) {
                if (bConnect(feamid, filid)) {
                    if (m_ptr == NULL) {
                        prevLOutput->at(feamid)->convolution(m_ptr,
                                                             filterArr[filid],
                                                             config.padding, config.stride);
                        assert(m_ptr);
                    } else {
                        prevLOutput->at(feamid)->convolution(cache,
                                                             filterArr[filid],
                                                             config.padding, config.stride);
                        assert(cache);
                        m_ptr->add(cache);
                    }
                    
                }
            }
            if (bias[filid] == NULL) { // Asynchronous to lazy init
                unique_lock<SpinLock> glock(this->lock);
                if (bias[filid] == NULL) { // double check
                    bias[filid] = new Matrix(m_ptr->x_len, m_ptr->y_len);
                    bias[filid]->randomInit();
                    biasDelta[filid] = new Matrix(m_ptr->x_len, m_ptr->y_len);
                    biasDelta[filid]->zeroInit();
                }
            }
            m_ptr->add(bias[filid]);
            
            // apply Activation Function
            m_ptr->operate([&](vector<double>* matrix) {
                assert(matrix);
                this->getActiveFun().forward(matrix);
            });
            output_act->at(filid) = m_ptr;
        }
        return this->nextLayer->forward(output_act);
    }
    
    void backward(vector<Matrix*>* outputDelta) {
        assert(outputDelta->size() == this->output_dimention);
        const vector<Matrix*> *prev_output_act = NULL;
        
        // init ThreadLocal var
        vector<Matrix*>*& input_delta = *tl_input_delta;
        if (input_delta == NULL) {
            input_delta = new vector<Matrix*>();
            input_delta->resize(this->input_dimention);
        }
        Matrix*& cache_bp = *tl_cache_bp;

        if (!this->bInputLayer) {
            assert(this->prevLayer);
            prev_output_act = this->prevLayer->output();
            
            FOR(i, this->input_dimention) {
                auto m_ptr = input_delta->at(i);
                if (m_ptr) {
                    m_ptr->zeroInit();
                }
                FOR(j, this->output_dimention) {
                    if (bConnect(i, j)) {
                        // delta Z_(L) conv rot180 W_(L) * di-acti( Z_(L-1) )
                        if (m_ptr == NULL) {
                            outputDelta->at(j)->deconvolution_Delta(m_ptr, filterArr[j],
                                                                    config.padding, config.stride);
                            assert(m_ptr);
                        } else {
                            outputDelta->at(j)->deconvolution_Delta(cache_bp, filterArr[j],
                                                                    config.padding, config.stride);
                            assert(cache_bp);
                            m_ptr->add(cache_bp);
                        }
                    }
                }
                m_ptr->operate([&, i](vector<double>* matrix) {
                    this->prevLayer->getActiveFun().backward(matrix,
                            this->prevLayer->output()->at(i)->pointer(), matrix);
                });
                input_delta->at(i) = m_ptr;
            }
            this->prevLayer->backward(input_delta);
        } else {
//            printf("Backward complete.\n");
        }
        
        // Asynchronous update filter weight and bias to minimize delta
        {
            unique_lock<SpinLock> glock(this->lock);
            FOR(filid, filter_cnt) {
                FOR(feamid, this->input_dimention) {
                    if (bConnect(feamid, filid)) {
                        // delta Z_(L) conv acti( Z_(L-1) )
                        if (this->bInputLayer) {
                            vector<Matrix*>*& input = *tl_input;
                            assert(input);
                            outputDelta->at(filid)->deconvolution_Filter(filterDelta[filid],
                                    input->at(feamid), config.padding, config.stride);
                        } else {
                            assert(prev_output_act != NULL);
                            outputDelta->at(filid)->deconvolution_Filter(filterDelta[filid],
                                    this->prevLayer->output()->at(feamid),
                                    config.padding, config.stride);
                        }
                    }
                }
                biasDelta[filid]->add(outputDelta->at(filid));
            }
        }
    }
    
    const vector<Matrix*>* output() {
        vector<Matrix*>*& output_act = *tl_output_act;
        assert(output_act);
        return output_act;
    }
    
    void applyBatchGradient() {
        updater.update(0, bias, biasDelta);
        updater.update(this->output_dimention, filterArr, filterDelta);
        
        if (this->nextLayer) {
            this->nextLayer->applyBatchGradient();
        }
    }
    
    void debugPrintFilter() {
        cout << "Conv Layer " << filter_cnt << " filters " << endl;
        FOR(filid, filter_cnt) {
            filterArr[filid]->debugPrint();
        }
    }
    
protected:
    bool bConnect(size_t input_fm_idx, size_t filter_idx) {
        if (this->input_dimention == 6 && this->output_dimention == 16) {
            assert(input_fm_idx * 16 + filter_idx < 6 * 16);
            return *(cnn_dropout_mask + input_fm_idx * 16 + filter_idx);
        }
        return true;
    }
    
    size_t filter_cnt;
    CNN_Config config;
    
    vector<Matrix*> filterArr;
    vector<Matrix*> bias;
    
    vector<Matrix*> filterDelta;
    vector<Matrix*> biasDelta;
    
    ThreadLocal<vector<Matrix*>*> tl_output_act;
    ThreadLocal<vector<Matrix*>*> tl_input_delta;
    ThreadLocal<vector<Matrix*>*> tl_input;
    
    ThreadLocal<Matrix*> tl_cache, tl_cache_bp;
    
    AdagradUpdater updater;
};

#endif /* convLayer_h */
