//
//  train_ffm_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/19.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_ffm_algo.h"
#include "../common/avx.h"

void Train_FFM_Algo::init() {
    L2Reg_ratio = 0.001f;
    
    learnable_params_cnt = this->feature_cnt * this->field_cnt * this->factor_cnt
                           + this->feature_cnt;
    update_g = new float[learnable_params_cnt];
    updater.learnable_params_cnt(learnable_params_cnt);
    
    printf("Training FFM\n");
}

void Train_FFM_Algo::Train() {
    
    GradientUpdater::__global_bTraining = true;
    GradientUpdater::__global_minibatch_size = dataRow_cnt;
    
    for (size_t i = 0; i < this->epoch; i++) {
        __loss = 0;
        __accuracy = 0;
        
        this->proc_data_left = (int)this->dataRow_cnt;
        
        size_t thread_hold_dataRow_cnt = (this->dataRow_cnt + this->proc_cnt - 1) / this->proc_cnt;
        
        for (size_t pid = 0; pid < this->proc_cnt; pid++) {
            size_t start_pos = pid * thread_hold_dataRow_cnt;
            threadpool->addTask(bind(&Train_FFM_Algo::batchGradCompute, this, start_pos,
                                     min(start_pos + thread_hold_dataRow_cnt, this->dataRow_cnt)));
        }
        threadpool->wait();
        
        printf("Epoch %zu Train Loss = %f Accuracy = %f\n", i, __loss, __accuracy / dataRow_cnt);
        // apply gradient
        ApplyGrad();
    }
    
    GradientUpdater::__global_bTraining = false;
}

void Train_FFM_Algo::batchGradCompute(size_t rbegin, size_t rend) {
    for (size_t rid = rbegin; rid < rend; rid++) { // data row
        float fm_pred = 0.0f;
        
        for (size_t i = 0; i < dataSet[rid].size(); i++) {
            const size_t fid = dataSet[rid][i].first;
            const float X = dataSet[rid][i].second;
            const size_t field = dataSet[rid][i].field;
            
            fm_pred += W[fid] * X;
            
            for (size_t j = i + 1; j < dataSet[rid].size(); j++) {
                const size_t fid2 = dataSet[rid][j].first;
                const float X2 = dataSet[rid][j].second;
                const size_t field2 = dataSet[rid][j].field;
                
                float field_w = avx_dotProduct(getV_field(fid, field2, 0),
                                               getV_field(fid2, field, 0), factor_cnt);
                fm_pred += field_w * X * X2;
            }
        }
        accumWVGrad(rid, sigmoid.forward(fm_pred));
    }
    assert(this->proc_data_left > 0);
    this->proc_data_left -= rend - rbegin;
}

void Train_FFM_Algo::accumWVGrad(size_t rid, float pred) {
    const float target = label[rid];
    const float loss = pred - target;
    if (loss == 0) {
        return;
    }
    __loss += target == 1 ? -log(pred) : -log(1.0 - pred);
    if (pred > 0.5 && target == 1) {
        __accuracy++;
    } else if (pred < 0.5 && target == 0) {
        __accuracy++;
    }
    
    size_t fid, fid2, field, field2;
    float x, x2;
    for (size_t i = 0; i < dataSet[rid].size(); i++) {
        fid = dataSet[rid][i].first;
        x = dataSet[rid][i].second;
        field = dataSet[rid][i].field;
        
        *update_W(fid) += loss * x + L2Reg_ratio * W[fid];
        
        for (size_t j = i + 1; j < dataSet[rid].size(); j++) {
            fid2 = dataSet[rid][j].first;
            x2 = dataSet[rid][j].second;
            field2 = dataSet[rid][j].field;

            const float scaler = x * x2 * loss;
            const float* v1 = getV_field(fid, field2, 0);
            const float* v2 = getV_field(fid2, field, 0);
            float* update_v1 = update_V(fid, field2, 0);
            float* update_v2 = update_V(fid2, field, 0);
            
            avx_vecScalerAdd(update_v1, v2, update_v1, scaler, factor_cnt);
            avx_vecScalerAdd(update_v1, v1, update_v1, L2Reg_ratio, factor_cnt);
            
            avx_vecScalerAdd(update_v2, v1, update_v2, scaler, factor_cnt);
            avx_vecScalerAdd(update_v2, v2, update_v2, L2Reg_ratio, factor_cnt);
        }
    }
}

void Train_FFM_Algo::ApplyGrad() {
    updater.update(0, this->feature_cnt, W, update_g);
    
    float *gradV = update_g + this->feature_cnt;
    updater.update(this->feature_cnt, this->feature_cnt *
                   this->field_cnt * this->factor_cnt, V, gradV);
}
