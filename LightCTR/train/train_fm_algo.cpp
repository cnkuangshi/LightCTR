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
    memset(sumVX, 0, sizeof(float) * this->dataRow_cnt * this->factor_cnt);
    
    update_g = new float[learnable_params_cnt];
    updater.learnable_params_cnt(learnable_params_cnt);
    
    update_threadLocal = new vector<float>*[this->proc_cnt];
    memset(update_threadLocal, NULL, sizeof(vector<float>*) * this->proc_cnt);
}

void Train_FM_Algo::flash() {
    memset(update_g, 0, sizeof(float) * learnable_params_cnt);
#ifdef FM
    memset(sumVX, 0, sizeof(float) * dataRow_cnt * factor_cnt);
#endif
}

void Train_FM_Algo::Train() {
    
    GradientUpdater::__global_bTraining = true;
    
    for (size_t i = 0; i < this->epoch_cnt; i++) {
        
        flash();
        this->proc_data_left = (int)this->dataRow_cnt;
        
        size_t thread_hold_dataRow_cnt = (this->dataRow_cnt + this->proc_cnt - 1) / this->proc_cnt;
        
        for (size_t pid = 0; pid < this->proc_cnt; pid++) {
            size_t start_pos = pid * thread_hold_dataRow_cnt;
            threadpool->addTask(bind(&Train_FM_Algo::batchGradCompute, this, pid, start_pos,
                                     min(start_pos + thread_hold_dataRow_cnt, this->dataRow_cnt)));
        }
        threadpool->wait();
        assert(proc_data_left == 0);
        
        ApplyGrad();
    }
    
    GradientUpdater::__global_bTraining = false;
}

void Train_FM_Algo::batchGradCompute(size_t pid, size_t rbegin, size_t rend) {

    if (update_threadLocal[pid] == NULL) {
        update_threadLocal[pid] = new vector<float>();
        update_threadLocal[pid]->resize(learnable_params_cnt);
    }
    fill(update_threadLocal[pid]->begin(), update_threadLocal[pid]->end(), 0.0f);
    
    vector<float> tmp_vec;
    tmp_vec.resize(factor_cnt);
    
    for (size_t rid = rbegin; rid < rend; rid++) { // data row
        float fm_pred = 0.0f;
        for (size_t i = 0; i < dataSet[rid].size(); i++) {
            const size_t fid = dataSet[rid][i].first;
            assert(fid < this->feature_cnt);
            
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
        accumWVGrad(rid, sigmoid.forward(fm_pred), update_threadLocal[pid]);
    }
    
    // synchronize to accumulate global gradient update_W and update_V
    {
        unique_lock<SpinLock> glock(this->lock);
        avx_vecAdd(update_W(0), update_threadLocal[pid]->data(), update_W(0), feature_cnt);
        for (size_t fid = 0; fid < this->feature_cnt; fid++) {
#ifdef FM
            avx_vecAdd(update_V(fid, 0),
                       update_threadLocal[pid]->data() + feature_cnt + fid * factor_cnt,
                       update_V(fid, 0), factor_cnt);
#endif
        }
        assert(this->proc_data_left > 0);
        this->proc_data_left -= rend - rbegin;
    }
}

void Train_FM_Algo::accumWVGrad(size_t rid, float pred, vector<float>* update_local) {
    assert(update_local && update_local->size() == learnable_params_cnt);
    const float target = label[rid];
    size_t fid;
    float x;
    vector<float> tmp_vec;
    tmp_vec.resize(factor_cnt);
    
    for (size_t i = 0; i < dataSet[rid].size(); i++) {
        fid = dataSet[rid][i].first;
        assert(fid < this->feature_cnt);
        x = dataSet[rid][i].second;
        const float gradW = LogisticGradW(pred, target, x) + L2Reg_ratio * W[fid];
        update_local->at(fid) += gradW;
#ifdef FM
        float* ptr = update_local->data() + feature_cnt + fid * factor_cnt;
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
