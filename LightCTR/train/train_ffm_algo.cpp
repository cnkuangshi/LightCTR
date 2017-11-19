//
//  train_ffm_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/19.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_ffm_algo.h"

void Train_FFM_Algo::init() {
    L2Reg_ratio = 0.001f;
    batch_size = GradientUpdater::__global_minibatch_size;
    
    learnable_params_cnt = this->feature_cnt * this->field_cnt * this->factor_cnt
                           + this->feature_cnt;
    update_g = new double[learnable_params_cnt];
    updater.learnable_params_cnt(learnable_params_cnt);
    
    printf("Training FFM\n");
}

void Train_FFM_Algo::Train() {
    
    GradientUpdater::__global_bTraining = true;
    
    for (size_t i = 0; i < this->epoch; i++) {
        
        size_t minibatch_epoch = (this->dataRow_cnt + this->batch_size - 1) / this->batch_size;
        
        for (size_t p = 0; p < minibatch_epoch; p++) {
            
            memset(update_g, 0, sizeof(double) * learnable_params_cnt);
            
            size_t start_pos = p * batch_size;
            batchGradCompute(start_pos, min(start_pos + batch_size, this->dataRow_cnt));
            // apply gradient
            ApplyGrad();
        }
    }
    
    GradientUpdater::__global_bTraining = false;
}

void Train_FFM_Algo::batchGradCompute(size_t rbegin, size_t rend) {
    threadpool->init();
    for (size_t rid = rbegin; rid < rend; rid++) { // data row
        threadpool->addTask([&, rid]() {
            double fm_pred = 0.0f;
            
            for (size_t i = 0; i < dataSet[rid].size(); i++) {
                size_t fid = dataSet[rid][i].first;
                assert(fid < this->feature_cnt);
                double X = dataSet[rid][i].second;
                size_t field = dataSet[rid][i].field;
                
                fm_pred += W[fid] * X;
                
                for (size_t j = i + 1; j < dataSet[rid].size(); j++) {
                    size_t fid2 = dataSet[rid][i].first;
                    double X2 = dataSet[rid][i].second;
                    size_t field2 = dataSet[rid][i].field;
                    
                    double field_pred = 0;
                    for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
                        double v1 = *getV_field(fid, field2, fac_itr);
                        double v2 = *getV_field(fid2, field, fac_itr);
                        field_pred += v1 * v2;
                    }
                    fm_pred += field_pred * X * X2;
                }
            }
            accumWVGrad(rid, sigmoid.forward(fm_pred));
        });
    }
    threadpool->join();
}

void Train_FFM_Algo::accumWVGrad(size_t rid, double pred) {
    double target = label[rid];
    size_t fid, fid2, x, x2, field, field2;
    for (size_t i = 0; i < dataSet[rid].size(); i++) {
        if (dataSet[rid][i].second == 0 || pred - target == 0) {
            continue;
        }
        fid = dataSet[rid][i].first;
        x = dataSet[rid][i].second;
        field = dataSet[rid][i].field;
        {
            unique_lock<mutex> glock(this->lock_w);
            *update_W(fid) += (pred - target) * x + L2Reg_ratio * W[fid];;
        }
        
        for (size_t j = i + 1; j < dataSet[rid].size(); j++) {
            fid2 = dataSet[rid][i].first;
            x2 = dataSet[rid][i].second;
            field2 = dataSet[rid][i].field;
            
            x2 *= x;
            {
                unique_lock<mutex> glock(this->lock_v);
                for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
                    double v1 = *getV_field(fid, field2, fac_itr);
                    double v2 = *getV_field(fid2, field, fac_itr);
                    *update_V(fid, field2, fac_itr) += (pred - target) * x2 * v2 + L2Reg_ratio * v1;
                    *update_V(fid2, field, fac_itr) += (pred - target) * x2 * v1 + L2Reg_ratio * v2;
                }
            }
        }
    }
}

void Train_FFM_Algo::ApplyGrad() {
    updater.update(0, this->feature_cnt, W, update_g);
    
    double *gradV = update_g + this->feature_cnt;
    updater.update(this->feature_cnt, this->feature_cnt *
                   this->field_cnt * this->factor_cnt, V, gradV);
}
