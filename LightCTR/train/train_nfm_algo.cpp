//
//  train_nfm_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/6.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_nfm_algo.h"

void Train_NFM_Algo::init() {
    L2Reg_ratio = 0.001f;
    batch_size = GradientUpdater::__global_minibatch_size;
    
    learnable_params_cnt = this->feature_cnt * (this->factor_cnt + 1);
    update_g = new float[learnable_params_cnt];
    updater.learnable_params_cnt(learnable_params_cnt);
    
    sumVX = new float[this->dataRow_cnt * this->factor_cnt];
    memset(sumVX, 0, sizeof(float) * this->dataRow_cnt * this->factor_cnt);
    
    printf("-- Inner FC-1 ");
    this->inputLayer = new Fully_Conn_Layer<Sigmoid>(NULL, this->factor_cnt,
                                                     this->hidden_layer_size);
    this->inputLayer->needInputDelta = true;
    printf("-- Inner FC-2 ");
    this->outputLayer = new Fully_Conn_Layer<Sigmoid>(inputLayer, this->hidden_layer_size, 1);
}

void Train_NFM_Algo::Train() {
    
    GradientUpdater::__global_bTraining = true;
    
    for (size_t i = 0; i < this->epoch; i++) {
        
        loss = 0;
        accuracy = 0;
        memset(sumVX, 0, sizeof(float) * this->dataRow_cnt * this->factor_cnt);
        
        size_t minibatch_epoch = (this->dataRow_cnt + this->batch_size - 1) / this->batch_size;
        
        for (size_t p = 0; p < minibatch_epoch; p++) {
            memset(update_g, 0, sizeof(float) * learnable_params_cnt);
            
            size_t start_pos = p * batch_size;
            batchGradCompute(start_pos, min(start_pos + batch_size, this->dataRow_cnt));
            // apply gradient
            ApplyGrad();
        }
        printf("Epoch %zu loss = %f accuracy = %f\n", i, loss, 1.0 * accuracy / dataRow_cnt);
    }
    
    GradientUpdater::__global_bTraining = false;
}

void Train_NFM_Algo::batchGradCompute(size_t rbegin, size_t rend) {
    for (size_t rid = rbegin; rid < rend; rid++) { // data row
        threadpool->addTask([&, rid]() {
            // init threadlocal var
            Matrix*& fc_input_Matrix = *tl_fc_input_Matrix;
            if (fc_input_Matrix == NULL) {
                fc_input_Matrix = new Matrix(1, factor_cnt);
            }
            Matrix*& fc_bp_Matrix = *tl_fc_bp_Matrix;
            if (fc_bp_Matrix == NULL) {
                fc_bp_Matrix = new Matrix(1, 1);
            }
            vector<Matrix*>*& wrapper = *tl_wrapper;
            if (wrapper == NULL) {
                wrapper = new vector<Matrix*>();
                wrapper->resize(1);
            }
            
            fc_input_Matrix->zeroInit();
            float fm_pred = 0.0f;
            
            vector<float> tmp_vec, tmp_vec2;
            tmp_vec.resize(factor_cnt);
            tmp_vec2.resize(factor_cnt);
            
            for (size_t i = 0; i < dataSet[rid].size(); i++) {
                const size_t fid = dataSet[rid][i].first;
                assert(fid < this->feature_cnt);
                
                const float X = dataSet[rid][i].second;
                fm_pred += W[fid] * X; // wide part
                
                avx_vecScale(getV(fid, 0), tmp_vec.data(), factor_cnt, X);
                avx_vecAdd(getSumVX(rid, 0), tmp_vec.data(), getSumVX(rid, 0), factor_cnt);
                avx_vecScale(tmp_vec.data(), tmp_vec2.data(), factor_cnt, -0.5);
                avx_vecScalerAdd(fc_input_Matrix->getEle(0, 0),
                                 tmp_vec.data(),
                                 fc_input_Matrix->getEle(0, 0),
                                 tmp_vec2.data(), factor_cnt);
            }
            avx_vecScale(getSumVX(rid, 0), tmp_vec.data(), factor_cnt, 0.5);
            avx_vecScalerAdd(fc_input_Matrix->getEle(0, 0), getSumVX(rid, 0), fc_input_Matrix->getEle(0, 0), tmp_vec.data(), factor_cnt);
            
            // deep part
            wrapper->at(0) = fc_input_Matrix;
            const vector<float> *fc_pred = this->inputLayer->forward(wrapper);
            assert(fc_pred && fc_pred->size() == 1);

            fm_pred += fc_pred->at(0);
            fm_pred = sigmoid.forward(fm_pred); // FM activate
            
            loss += (int)label[rid] == 1 ? -log(fm_pred) : -log(1.0 - fm_pred);
            if (fm_pred > 0.5 && label[rid] == 1) {
                accuracy++;
            } else if (fm_pred < 0.5 && label[rid] == 0) {
                accuracy++;
            }
            
            // accumulate grad
            accumWideGrad(rid, fm_pred);
            
            // FC backward
            *fc_bp_Matrix->getEle(0, 0) = fm_pred - label[rid];
            wrapper->at(0) = fc_bp_Matrix;
            this->outputLayer->backward(wrapper);
            const Matrix* delta = this->inputLayer->inputDelta(); // get delta of VX
            assert(delta->size() == factor_cnt);
            accumDeepGrad(rid, delta->pointer());
        });
    }
    threadpool->wait();
}

void Train_NFM_Algo::accumWideGrad(size_t rid, float pred) {
    const float target = label[rid];
    size_t fid;
    float x;
    for (size_t i = 0; i < dataSet[rid].size(); i++) {
        fid = dataSet[rid][i].first;
        assert(fid < this->feature_cnt);
        x = dataSet[rid][i].second;
        
        *update_W(fid) += LogisticGradW(pred, target, x) + L2Reg_ratio * W[fid];
    }
}

void Train_NFM_Algo::accumDeepGrad(size_t rid, vector<float>* delta) {
    size_t fid;
    float X;
    
    vector<float> tmp_vec;
    tmp_vec.resize(factor_cnt);
    
    for (size_t i = 0; i < dataSet[rid].size(); i++) {
        fid = dataSet[rid][i].first;
        assert(fid < this->feature_cnt);
        X = dataSet[rid][i].second;

        avx_vecScalerAdd(getSumVX(rid, 0), getV(fid, 0),
                         tmp_vec.data(), -X, factor_cnt);
        avx_vecScale(delta->data(), delta->data(), factor_cnt, X);
        avx_vecScalerAdd(update_V(fid, 0), tmp_vec.data(),
                         update_V(fid, 0), delta->data(), factor_cnt);
        avx_vecScalerAdd(update_V(fid, 0), getV(fid, 0), update_V(fid, 0), L2Reg_ratio, factor_cnt);
    }
}

void Train_NFM_Algo::ApplyGrad() {
    // update wide part
    updater.update(0, this->feature_cnt, W, update_g);
    // update v deep part
    float *gradV = update_g + this->feature_cnt;
    updater.update(this->feature_cnt, this->feature_cnt * this->factor_cnt, V, gradV);
    // update fc deep part
    this->inputLayer->applyBatchGradient();
}

