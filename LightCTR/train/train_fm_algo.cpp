//
//  train_fm_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/23.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_fm_algo.h"
#include "../common/avx.h"

void Train_FM_Algo::init() {
    L2Reg_ratio = 0.001f;
#ifdef FM
    learnable_params_cnt = this->feature_cnt * (this->factor_cnt + 1);
#else
    learnable_params_cnt = this->feature_cnt;
#endif
    sumVX = new float[this->dataRow_cnt * this->factor_cnt];
    assert(sumVX);
    memset(sumVX, 0, sizeof(float) * this->dataRow_cnt * this->factor_cnt);
    
    update_g = new float[learnable_params_cnt];
    assert(update_g);
    updater.learnable_params_cnt(learnable_params_cnt);
}

void Train_FM_Algo::flash() {
    memset(update_g, 0, sizeof(float) * learnable_params_cnt);
#ifdef FM
    memset(sumVX, 0, sizeof(float) * dataRow_cnt * factor_cnt);
#endif
}

void Train_FM_Algo::Train() {
    
    GradientUpdater::__global_bTraining = true;
    GradientUpdater::__global_minibatch_size = dataRow_cnt;
    
    for (size_t i = 0; i < this->epoch_cnt; i++) {
        __loss = 0;
        __accuracy = 0;
        
        flash();
        this->proc_data_left = (int)this->dataRow_cnt;
        
        size_t thread_hold_dataRow_cnt = (this->dataRow_cnt + this->proc_cnt - 1) / this->proc_cnt;
        
        for (size_t pid = 0; pid < this->proc_cnt; pid++) {
            size_t start_pos = pid * thread_hold_dataRow_cnt;
            threadpool->addTask(bind(&Train_FM_Algo::batchGradCompute, this, start_pos,
                                     min(start_pos + thread_hold_dataRow_cnt, this->dataRow_cnt)));
        }
        threadpool->wait();
        assert(proc_data_left == 0);
        
        printf("Epoch %zu Train Loss = %f Accuracy = %f\n", i, __loss, __accuracy / dataRow_cnt);
        ApplyGrad();
    }
    
    GradientUpdater::__global_bTraining = false;
}

void Train_FM_Algo::batchGradCompute(size_t rbegin, size_t rend) {
    
    vector<float> tmp_vec;
    tmp_vec.resize(factor_cnt);
    
    for (size_t rid = rbegin; rid < rend; rid++) { // data row
        float fm_pred = 0.0f;
        for (size_t i = 0; i < dataSet[rid].size(); i++) {
            const size_t fid = dataSet[rid][i].first;
            
            const float X = dataSet[rid][i].second;
            fm_pred += W[fid] * X;
#ifdef FM
            avx_vecScale(getV(fid, 0), tmp_vec.data(), factor_cnt, X);
            avx_vecAdd(getSumVX(rid, 0), tmp_vec.data(), getSumVX(rid, 0), factor_cnt);
            fm_pred -= 0.5 * avx_dotProduct(tmp_vec.data(), tmp_vec.data(), factor_cnt);
#endif
        }
#ifdef FM
        fm_pred += 0.5 * avx_dotProduct(getSumVX(rid, 0), getSumVX(rid, 0), factor_cnt);
#endif
        accumWVGrad(rid, sigmoid.forward(fm_pred));
    }
    
    this->proc_data_left -= rend - rbegin;
}

void Train_FM_Algo::accumWVGrad(size_t rid, float pred) {
    const float target = label[rid];
    
    __loss += target == 1 ? -log(pred) : -log(1.0 - pred);
    if (pred > 0.5 && target == 1) {
        __accuracy++;
    } else if (pred < 0.5 && target == 0) {
        __accuracy++;
    }
    
    size_t fid;
    float x;
    vector<float> tmp_vec;
    tmp_vec.resize(factor_cnt);
    
    for (size_t i = 0; i < dataSet[rid].size(); i++) {
        fid = dataSet[rid][i].first;
        x = dataSet[rid][i].second;
        const float gradW = LogisticGradW(pred, target, x) + L2Reg_ratio * W[fid];
        *update_W(fid) += gradW;
#ifdef FM
        float* ptr = update_V(fid, 0);
        avx_vecScalerAdd(getSumVX(rid, 0), getV(fid, 0),
                         tmp_vec.data(), -x, factor_cnt);
        avx_vecScalerAdd(ptr, tmp_vec.data(), ptr, gradW, factor_cnt);
        avx_vecScalerAdd(ptr, getV(fid, 0), ptr, L2Reg_ratio, factor_cnt);
#endif
    }
}

void Train_FM_Algo::ApplyGrad() {
    
    updater.update(0, this->feature_cnt, W, update_g);
#ifdef FM
    float *gradV = update_g + this->feature_cnt;
    updater.update(this->feature_cnt, this->feature_cnt * this->factor_cnt, V, gradV);
#endif
}
