//
//  train_fm_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/23.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_fm_algo.h"
#include "unistd.h"

void Train_FM_Algo::init() {
    L2Reg_ratio = 0.001f;
#ifdef FM
    learnable_params_cnt = this->feature_cnt * (this->factor_cnt + 1);
#else
    learnable_params_cnt = this->feature_cnt;
#endif
    sumVX = new double[this->dataRow_cnt * this->factor_cnt];
    memset(sumVX, 0, sizeof(double) * this->dataRow_cnt * this->factor_cnt);
    
    printf("Training FM dropout = %.2f\n", dropout.dropout_rate);
    
    update_g = new double[learnable_params_cnt];
    updater.learnable_params_cnt(learnable_params_cnt);
    
    update_threadLocal = new vector<double>*[this->proc_cnt];
    memset(update_threadLocal, NULL, sizeof(vector<double>*) * this->proc_cnt);
}

void Train_FM_Algo::flash() {
    memset(update_g, 0, sizeof(double) * learnable_params_cnt);
#ifdef FM
    memset(sumVX, 0, sizeof(double) * dataRow_cnt * factor_cnt);
#endif
}

void Train_FM_Algo::Train() {
    
    GradientUpdater::__global_bTraining = true;
    
    for (size_t i = 0; i < this->epoch_cnt; i++) {
        
        flash();
        this->proc_data_left = (int)this->dataRow_cnt;
        
        size_t thread_hold_dataRow_cnt = (this->dataRow_cnt + this->proc_cnt - 1) / this->proc_cnt;
        
        for (size_t pid = 0; pid < this->proc_cnt; pid++) {
            // re-sample dropout
            dropout.Mask(dropout_mask, this->factor_cnt);
            
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
        update_threadLocal[pid] = new vector<double>();
        update_threadLocal[pid]->resize(learnable_params_cnt);
    }
    fill(update_threadLocal[pid]->begin(), update_threadLocal[pid]->end(), 0.0f);
    
    for (size_t rid = rbegin; rid < rend; rid++) { // data row
        double fm_pred = 0.0f;
        for (size_t i = 0; i < dataSet[rid].size(); i++) {
            const size_t fid = dataSet[rid][i].first;
            assert(fid < this->feature_cnt);
            
            const double X = dataSet[rid][i].second;
            fm_pred += W[fid] * X * dropout.rescale();
#ifdef FM
            for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
                if (!dropout_mask[fac_itr]) { // apply dropout mask
                    continue;
                }
                const double tmp = *getV(fid, fac_itr) * X;
                *getSumVX(rid, fac_itr) += tmp;
                fm_pred -= 0.5 * tmp * tmp * dropout.rescale();
            }
#endif
        }
#ifdef FM
        for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
            if (!dropout_mask[fac_itr]) { // apply dropout mask
                continue;
            }
            const double tmp = *getSumVX(rid, fac_itr);
            assert(!isnan(tmp));
            fm_pred += 0.5 * tmp * tmp * dropout.rescale();
        }
#endif
        accumWVGrad(rid, sigmoid.forward(fm_pred), update_threadLocal[pid]);
    }
    
    // synchronize to accumulate global gradient update_W and update_V
    {
        unique_lock<SpinLock> glock(this->lock);
        for (size_t fid = 0; fid < this->feature_cnt; fid++) {
            *update_W(fid) += update_threadLocal[pid]->at(fid);
#ifdef FM
            for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
                if (!dropout_mask[fac_itr]) { // apply dropout mask
                    continue;
                }
                *update_V(fid, fac_itr) +=
                    update_threadLocal[pid]->at(this->feature_cnt + fid * factor_cnt + fac_itr);
            }
#endif
        }
        assert(this->proc_data_left > 0);
        this->proc_data_left -= rend - rbegin;
    }
}

void Train_FM_Algo::accumWVGrad(size_t rid, double pred, vector<double>* update_local) {
    assert(update_local && update_local->size() == learnable_params_cnt);
    const double target = label[rid];
    size_t fid, x;
    for (size_t i = 0; i < dataSet[rid].size(); i++) {
        if (dataSet[rid][i].second == 0) {
            continue;
        }
        fid = dataSet[rid][i].first;
        assert(fid < this->feature_cnt);
        x = dataSet[rid][i].second;
        const double gradW = LogisticGradW(pred, target, x) + L2Reg_ratio * W[fid];
        update_local->at(fid) += gradW;
#ifdef FM
        for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
            if (!dropout_mask[fac_itr]) { // apply dropout mask
                continue;
            }
            const double sum = *getSumVX(rid, fac_itr);
            const double v = *getV(fid, fac_itr);
            update_local->at(this->feature_cnt + fid * factor_cnt + fac_itr)
                    += LogisticGradV(gradW, sum, v, x) + L2Reg_ratio * v;
        }
#endif
    }
}

void Train_FM_Algo::ApplyGrad() {
    GradientUpdater::__global_minibatch_size = dataRow_cnt;
    
    updater.update(0, this->feature_cnt, W, update_g);
#ifdef FM
    double *gradV = update_g + this->feature_cnt;
    updater.update(this->feature_cnt, this->feature_cnt * this->factor_cnt, V, gradV);
#endif
}
