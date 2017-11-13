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
    update_g = new double[learnable_params_cnt];
    updater.learnable_params_cnt(learnable_params_cnt);
    
    printf("Training NFM dropout = %.2f\n", dropout.dropout_rate);
    
    printf("-- Inner FC-1 ");
    this->inputLayer = new Fully_Conn_Layer<Sigmoid>(NULL, this->factor_cnt, this->hidden_layer_size);
    this->inputLayer->needInputDelta = true;
    printf("-- Inner FC-2 ");
    this->outputLayer = new Fully_Conn_Layer<Sigmoid>(inputLayer, this->hidden_layer_size, 1);
}

void Train_NFM_Algo::Train() {
    
    GradientUpdater::__global_bTraining = true;
    
    for (size_t i = 0; i < this->epoch; i++) {
        
        memset(sumVX, 0, sizeof(double) * this->dataRow_cnt * this->factor_cnt);
        
        size_t minibatch_epoch = (this->dataRow_cnt + this->batch_size - 1) / this->batch_size;
        
        for (size_t p = 0; p < minibatch_epoch; p++) {
            // re-sample dropout
            dropout.Mask(dropout_mask, this->factor_cnt);
            
            memset(update_g, 0, sizeof(double) * learnable_params_cnt);
            
            size_t start_pos = p * batch_size;
            batchGradCompute(start_pos, min(start_pos + batch_size, this->dataRow_cnt));
            // apply gradient
            ApplyGrad();
        }
    }
    
    GradientUpdater::__global_bTraining = false;
}

void Train_NFM_Algo::batchGradCompute(size_t rbegin, size_t rend) {
    threadpool->init();
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
            double fm_pred = 0.0f;
            
            for (size_t i = 0; i < dataSet[rid].size(); i++) {
                size_t fid = dataSet[rid][i].first;
                assert(fid < this->feature_cnt);
                
                double X = dataSet[rid][i].second;
                fm_pred += W[fid] * X * dropout.rescale(); // wide part
                
                for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
                    if (!dropout_mask[fac_itr]) { // apply dropout mask
                        continue;
                    }
                    double tmp = *getV(fid, fac_itr) * X;
                    *getSumVX(rid, fac_itr) += tmp;
                    *fc_input_Matrix->getEle(0, fac_itr) -= 0.5 * tmp * tmp * dropout.rescale();
                }
            }
            for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
                if (!dropout_mask[fac_itr]) { // apply dropout mask
                    continue;
                }
                double tmp = *getSumVX(rid, fac_itr);
                assert(!isnan(tmp));
                *fc_input_Matrix->getEle(0, fac_itr) += 0.5 * tmp * tmp * dropout.rescale();
            }
            
            // deep part
            wrapper->at(0) = fc_input_Matrix;
            vector<double> *fc_pred = this->inputLayer->forward(wrapper);
            assert(fc_pred && fc_pred->size() == 1);

            fm_pred += fc_pred->at(0);
            fm_pred = sigmoid.forward(fm_pred); // FM activate
            
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
    threadpool->join();
}

void Train_NFM_Algo::accumWideGrad(size_t rid, double pred) {
    double target = label[rid];
    size_t fid, x;
    for (size_t i = 0; i < dataSet[rid].size(); i++) {
        if (dataSet[rid][i].second == 0) {
            continue;
        }
        fid = dataSet[rid][i].first;
        assert(fid < this->feature_cnt);
        x = dataSet[rid][i].second;
        
        {
            unique_lock<mutex> glock(this->lock_w);
            *update_W(fid) += LogisticGradW(pred, target, x) + L2Reg_ratio * W[fid];;
        }
    }
}

void Train_NFM_Algo::accumDeepGrad(size_t rid, vector<double>* delta) {
    size_t fid, X;
    for (size_t i = 0; i < dataSet[rid].size(); i++) {
        if (dataSet[rid][i].second == 0) {
            continue;
        }
        fid = dataSet[rid][i].first;
        assert(fid < this->feature_cnt);
        X = dataSet[rid][i].second;

        for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
            if (!dropout_mask[fac_itr]) { // apply dropout mask
                continue;
            }
            double grad = delta->at(fac_itr) * X;
            
            double sum = *getSumVX(rid, fac_itr);
            double v = *getV(fid, fac_itr);
            
            {
                unique_lock<mutex> glock(this->lock_v);
                *update_V(fid, fac_itr) += 0.1 * LogisticGradV(grad, sum, v, X) + L2Reg_ratio * v;
            }
        }
    }
}

void Train_NFM_Algo::ApplyGrad() {
    // update wide part
    updater.update(0, this->feature_cnt, W, update_g);
    // update v deep part
    double *gradV = update_g + this->feature_cnt;
    updater.update(this->feature_cnt, this->feature_cnt * this->factor_cnt, V, gradV);
    // update fc deep part
    this->inputLayer->applyBatchGradient();
}

