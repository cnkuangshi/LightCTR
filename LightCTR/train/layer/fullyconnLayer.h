//
//  fullyconnLayer.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/24.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef fullyconnLayer_h
#define fullyconnLayer_h

#include "layer_abst.h"
#include "../../util/matrix.h"

// Fully Connected Layer
template <typename ActivationFunction>
class Fully_Conn_Layer : public Layer_Base {
    
public:
    Fully_Conn_Layer(Layer_Base* _prevLayer, size_t _input_dimension, size_t _output_dimension):
    Layer_Base(_prevLayer, _input_dimension, _output_dimension) {
        init();
        printf("Fully Connected Layer\n");
    }
    Fully_Conn_Layer() = delete;
    
    ~Fully_Conn_Layer() {
        delete [] weight;
        delete [] bias;
        delete [] weightDelta;
        delete [] biasDelta;
        delete [] dropout_mask;
    }
    
    void init() {
        this->activeFun = new ActivationFunction();
        // allocate weight and bias's solver memory
        updater.learnable_params_cnt(this->output_dimension * (this->input_dimension + 1));
        
        needInputDelta = false;
        error_clip_threshold = 15;
        
        weight = new float[this->input_dimension * this->output_dimension];
        bias = new float[this->output_dimension];
        
        dropout_mask = new bool[this->output_dimension];
        
        FOR(i, this->output_dimension) {
            bias[i] = UniformNumRand() - 0.5f;
            dropout_mask[i] = SampleBinary(GradientUpdater::__global_sparse_rate);
            FOR(j, this->input_dimension) {
                *getWeight(i, j) = UniformNumRand() - 0.5f;
            }
        }
        
        // init for mini-batch
        weightDelta =  new float[this->input_dimension * this->output_dimension];
        memset(weightDelta, 0, this->input_dimension * this->output_dimension);
        biasDelta = new float[this->output_dimension];
        memset(biasDelta, 0, this->output_dimension);
    }
    
    void registerGradient(std::shared_ptr<BufferFusion<float> > _buf_fusion) {
        _buf_fusion->registMemChunk(weightDelta, this->input_dimension * this->output_dimension);
        _buf_fusion->registMemChunk(biasDelta, this->output_dimension);
        if (this->nextLayer) {
            this->nextLayer->registerGradient(_buf_fusion);
        }
    }
    
    // Attention fc layer's input vector only have one matrix to save memory,
    // so we need an adapter layer to convert between
    // matrix's vector[1x1, 1x1, ...N] and one matrix[N]
    vector<float>* forward(vector<Matrix*>* const prevLOutputMatrix) { // prevLOutput is acti( Z_(L-1) )
        assert(prevLOutputMatrix->size() == 1);
        vector<float>* prevLOutput = prevLOutputMatrix->at(0)->pointer();
        assert(prevLOutput->size() == this->input_dimension);
        
        // init ThreadLocal var
        Matrix*& output_act = *tl_output_act;
        if (output_act == NULL) {
            output_act = new Matrix(1, this->output_dimension);
        }
        
        if (this->bInputLayer) { // storage input only for input layer
            vector<float>*& input = *tl_input;
            if (input == NULL) {
                input = new vector<float>();
                input->resize(this->input_dimension);
            }
            input->assign(prevLOutput->begin(), prevLOutput->end());
        }
        
        FOR(i, this->output_dimension) {
            if (!dropout_mask[i] && this->nextLayer != NULL) { // apply dropout mask for output
                *output_act->getEle(0, i) = 0.0f;
                continue;
            }
            float sum = avx_dotProduct(prevLOutput->data(), getWeight(i, 0), input_dimension);
            sum += bias[i];
            *output_act->getEle(0, i) = sum;
            assert(!isnan(sum));
        }
        
        // init threadlocal wrapper
        vector<Matrix*>*& wrapper = *tl_wrapper;
        if (wrapper == NULL) {
            wrapper = new vector<Matrix*>();
            wrapper->resize(1);
        }
        
        if (this->nextLayer) {
            this->getActiveFun().forward(output_act->pointer());
            wrapper->at(0) = output_act;
            return this->nextLayer->forward(wrapper);
        } else {
//            printf("Forward complete.\n");
            return output_act->pointer(); // output layer return wx+b without activator
        }
    }
    
    void backward(vector<Matrix*>* const outputDeltaMatrix) { // outputDelta is Z_(L)
        vector<float>* outputDelta = outputDeltaMatrix->at(0)->pointer();
        assert(outputDelta->size() == this->output_dimension);
        
        // init ThreadLocal var
        Matrix*& input_delta = *tl_input_delta;
        if (input_delta == NULL) {
            input_delta = new Matrix(1, this->input_dimension);
        }
        
        // grad clipping for gradient explosion
        if (error_clip_threshold != 0) {
            outputDeltaMatrix->at(0)->clipping(error_clip_threshold);
        }
        
        vector<float>* prev_output_act = NULL;
        // Z_(L) = W_(L) * acti( Z_(L-1) ) + b
        if (!this->bInputLayer || needInputDelta) {
            FOR(i, this->input_dimension) {
                float sum = 0.0f;
                FOR(j, this->output_dimension) {
                    if (!dropout_mask[j] && this->nextLayer != NULL) {
                        // apply dropout mask for delta and re-scale
                        continue;
                    }
                    sum += outputDelta->at(j) * (*getWeight(j, i));
                }
                *input_delta->getEle(0, i) = sum;
            }
            if (!this->bInputLayer) {
                assert(this->prevLayer);
                prev_output_act = this->prevLayer->output()->at(0)->pointer();
                assert(prev_output_act && prev_output_act->size() == this->input_dimension);
                
                this->prevLayer->getActiveFun().backward(input_delta->pointer(), prev_output_act, input_delta->pointer());
                assert(input_delta->pointer()->size() == this->input_dimension);
                
                vector<Matrix*>*& wrapper = *tl_wrapper;
                assert(wrapper);
                wrapper->at(0) = input_delta;
                this->prevLayer->backward(wrapper);
            }
        } else {
//            printf("Backward complete.\n");
        }
        
        // Asynchronous update weight and bias to minimize delta
        {
            unique_lock<SpinLock> glock(this->lock);
            if (this->bInputLayer) {
                FOR(j, this->output_dimension) {
                    vector<float>*& input = *tl_input;
                    assert(input);
                    avx_vecScalerAdd(getWeightDelta(j, 0), input->data(),
                                     getWeightDelta(j, 0), outputDelta->at(j), input_dimension);
                }
            } else {
                FOR(j, this->output_dimension) {
                    assert(prev_output_act);
                    avx_vecScalerAdd(getWeightDelta(j, 0), prev_output_act->data(),
                                     getWeightDelta(j, 0), outputDelta->at(j), input_dimension);
                }
            }
            avx_vecAdd(biasDelta, outputDelta->data(), biasDelta, output_dimension);
        }
    }
    
    const vector<Matrix*>* output() {
        Matrix*& output_act = *tl_output_act;
        assert(output_act);
        
        vector<Matrix*>*& wrapper = *tl_wrapper;
        assert(wrapper);
        wrapper->at(0) = output_act;
        return wrapper;
    }
    const Matrix* inputDelta() {
        assert(needInputDelta);
        return *tl_input_delta;
    }
    
    void applyBatchGradient() {
        updater.update(0, this->output_dimension, bias, biasDelta);
        updater.update(this->output_dimension, this->output_dimension * this->input_dimension,
                       weight, weightDelta);
        
        if (this->nextLayer) {
            this->nextLayer->applyBatchGradient();
        }
    }
    
    bool needInputDelta;
    
protected:
    inline float* getWeight(size_t out_d, size_t in_d) const {
        return &weight[out_d * input_dimension + in_d];
    }
    inline float* getWeightDelta(size_t out_d, size_t in_d) const {
        return &weightDelta[out_d * input_dimension + in_d];
    }
    
    float* weight;
    float* bias;
    
    float* weightDelta;
    float* biasDelta;
    
    bool* dropout_mask;
    
    ThreadLocal<Matrix*> tl_output_act; // wx + b with activation
    ThreadLocal<Matrix*> tl_input_delta; // delta of prevLayer wx+b Z_(L-1)
    ThreadLocal<vector<float>*> tl_input;
    
    float error_clip_threshold;
    
    ThreadLocal<vector<Matrix*>*> tl_wrapper;
    
    AdagradUpdater_Num updater;
};

#endif /* fullyconnLayer_h */
