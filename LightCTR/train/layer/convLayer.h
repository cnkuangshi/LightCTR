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
    Conv_Layer(Layer_Base* _prevLayer, size_t _input_dimension,
               size_t _output_dimension, CNN_Config _config):
    Layer_Base(_prevLayer, _input_dimension, _output_dimension), config(_config) {
        init();
        printf("Convolution Layer\n");
    }
    Conv_Layer() = delete;
    
    ~Conv_Layer() {
        FOR(i, filter_cnt) {
            delete filterArr[i];
        }
        filterArr.clear();
        FOR(i, this->output_dimension) {
            delete bias[i];
        }
        bias.clear();
        FOR(i, filter_cnt) {
            delete filterDelta[i];
        }
        filterDelta.clear();
        FOR(i, filter_cnt) {
            delete biasDelta[i];
        }
        biasDelta.clear();
    }
    
    void init() {
        this->activeFun = new ActivationFunction();
        // allocate filter and bias's solver memory
        updater.learnable_params_cnt(this->output_dimension * 2);
        
        filter_cnt = this->output_dimension;
        
        filterArr.resize(filter_cnt);
        filterDelta.resize(filter_cnt);
        FOR(i, filter_cnt) {
            filterArr[i] = new Matrix(config.filter_size, config.filter_size);
            filterArr[i]->randomInit();
            filterDelta[i] = new Matrix(config.filter_size, config.filter_size);
            filterDelta[i]->zeroInit();
        }
        
        bias.resize(filter_cnt);
        biasDelta.resize(filter_cnt);
        FOR(i, filter_cnt) { // lazy init because they depend on conv result size
            bias[i] = NULL;
            biasDelta[i] = NULL;
        }
    }
    
    void registerInitializer(std::shared_ptr<BufferFusion<float> > _buf_fusion) {
        FOR(i, filter_cnt) {
            _buf_fusion->registMemChunk(filterArr[i]->pointer()->data(), filterArr[i]->size());
        }
        if (this->nextLayer) {
            this->nextLayer->registerInitializer(_buf_fusion);
        }
    }
    
    void registerGradient(std::shared_ptr<BufferFusion<float> > _buf_fusion) {
        FOR(i, filter_cnt) {
            _buf_fusion->registMemChunk(filterDelta[i]->pointer()->data(), filterDelta[i]->size());
            _buf_fusion->registMemChunk(biasDelta[i]->pointer()->data(), biasDelta[i]->size());
        }
        if (this->nextLayer) {
            this->nextLayer->registerGradient(_buf_fusion);
        }
    }
    
    vector<float>& forward(const vector<Matrix*>& prevLOutput) {
        assert(this->nextLayer);
        assert(prevLOutput.size() == this->input_dimension);
        
        // init ThreadLocal var
        MatrixArr& output_act = *tl_output_act;
        output_act.arr.resize(this->output_dimension);
        Matrix* cache = NULL;
        
        if (this->bInputLayer) { // storage input only for input layer
            vector<Matrix*>& input = *tl_input;
            input.resize(this->input_dimension);
            FOR(i, this->input_dimension) {
                input[i] = prevLOutput[i];
            }
        }
        
        FOR(filid, filter_cnt) {
            auto m_ptr = output_act.arr[filid];
            if (m_ptr) {
                m_ptr->zeroInit();
            }
            FOR(feamid, this->input_dimension) {
                if (bConnect(feamid, filid)) {
                    if (m_ptr == NULL) {
                        prevLOutput[feamid]->convolution(m_ptr,
                                                         filterArr[filid],
                                                         config.padding, config.stride);
                    } else {
                        prevLOutput[feamid]->convolution(cache,
                                                         filterArr[filid],
                                                         config.padding, config.stride);
                        assert(cache);
                        m_ptr->add(cache);
                    }
                }
            }
            if (bias[filid] == NULL) { // lazy init
                unique_lock<SpinLock> glock(this->lock);
                if (bias[filid] == NULL) { // double check
                    bias[filid] = new Matrix(m_ptr->x_len, m_ptr->y_len);
                    bias[filid]->zeroInit();
                    biasDelta[filid] = new Matrix(m_ptr->x_len, m_ptr->y_len);
                    biasDelta[filid]->zeroInit();
                }
            }
            m_ptr->add(bias[filid]);
            
            // apply Activation Function
            m_ptr->operate([this](vector<float>* matrix) {
                assert(matrix);
                this->getActiveFun().forward(matrix->data(), matrix->size());
            });
            output_act.arr[filid] = m_ptr;
        }
        delete cache;
        return this->nextLayer->forward(output_act.arr);
    }
    
    void backward(const vector<Matrix*>& outputDelta) {
        assert(outputDelta.size() == this->output_dimension);
        vector<Matrix*> prev_output_act;
        
        // init ThreadLocal var
        MatrixArr& input_delta = *tl_input_delta;
        input_delta.arr.resize(this->input_dimension);
        Matrix* cache_bp = NULL;

        if (!this->bInputLayer) {
            assert(this->prevLayer);
            prev_output_act = this->prevLayer->output();
            
            FOR(i, this->input_dimension) {
                auto m_ptr = input_delta.arr[i];
                if (m_ptr) {
                    m_ptr->zeroInit();
                }
                FOR(j, this->output_dimension) {
                    if (bConnect(i, j)) {
                        // delta Z_(L) conv rot180 W_(L) * di-acti( Z_(L-1) )
                        if (m_ptr == NULL) {
                            outputDelta[j]->deconvolution_Delta(m_ptr, filterArr[j],
                                                                config.padding, config.stride);
                        } else {
                            outputDelta[j]->deconvolution_Delta(cache_bp, filterArr[j],
                                                                config.padding, config.stride);
                            assert(cache_bp);
                            m_ptr->add(cache_bp);
                        }
                    }
                }
                m_ptr->operate([&, i](vector<float>* matrix) {
                    this->prevLayer->getActiveFun().backward(matrix->data(),
                                            this->prevLayer->output()[i]->pointer()->data(),
                                            matrix->data(), matrix->size());
                });
                input_delta.arr[i] = m_ptr;
            }
            delete cache_bp;
            this->prevLayer->backward(input_delta.arr);
        }
        
        // Asynchronous update filter weight and bias to minimize delta
        {
            unique_lock<SpinLock> glock(this->lock);
            FOR(filid, filter_cnt) {
                FOR(feamid, this->input_dimension) {
                    // delta Z_(L) conv acti( Z_(L-1) )
                    if (this->bInputLayer) {
                        vector<Matrix*>& input = *tl_input;
                        outputDelta[filid]->deconvolution_Filter(filterDelta[filid],
                                input[feamid], config.padding, config.stride);
                    } else {
                        outputDelta[filid]->deconvolution_Filter(filterDelta[filid],
                                this->prevLayer->output()[feamid],
                                config.padding, config.stride);
                    }
                }
                biasDelta[filid]->add(outputDelta[filid]);
            }
        }
    }
    
    const vector<Matrix*>& output() {
        MatrixArr& output_act = *tl_output_act;
        return output_act.arr;
    }
    
    void applyBatchGradient() {
        updater.update(0, bias, biasDelta);
        updater.update(this->output_dimension, filterArr, filterDelta);
        
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
        if (this->input_dimension == 6 && this->output_dimension == 16) {
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
    
    ThreadLocal<MatrixArr> tl_output_act;
    ThreadLocal<MatrixArr> tl_input_delta;
    ThreadLocal<vector<Matrix*> > tl_input;
    
    AdagradUpdater updater;
};

#endif /* convLayer_h */
