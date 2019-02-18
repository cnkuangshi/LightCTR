//
//  train_ffm_algo.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/19.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef train_ffm_algo_h
#define train_ffm_algo_h

#include "../fm_algo_abst.h"
#include <mutex>
#include <cmath>
#include "../util/activations.h"
#include "../util/gradientUpdater.h"
#include "../common/thread_pool.h"
#include "../common/lock.h"
using namespace std;

// Field-aware FM
class Train_FFM_Algo : public FM_Algo_Abst {
    
public:
    Train_FFM_Algo(string _dataPath, size_t _epoch_cnt,
                   size_t _factor_cnt, size_t _field_cnt):
    FM_Algo_Abst(_dataPath, _factor_cnt, _field_cnt), epoch(_epoch_cnt) {
        assert(this->feature_cnt != 0);
        threadpool = new ThreadPool(this->proc_cnt);
        init();
    }
    Train_FFM_Algo() = delete;
    
    ~Train_FFM_Algo() {
        delete threadpool;
        threadpool = NULL;
    }
    
    void init();
    void Train();
    
private:
    size_t epoch;
    size_t batch_size;
    
    Sigmoid sigmoid;
    
    size_t learnable_params_cnt;
    
    void batchGradCompute(size_t, size_t);
    void accumWVGrad(size_t rid, float pred);
    
    float *update_g;
    inline float* update_W(size_t fid) {
        return &update_g[fid];
    }
    inline float* update_V(size_t fid, size_t fieldid, size_t facid) {
        assert(this->feature_cnt + fid * this->field_cnt * this->factor_cnt
               + fieldid * this->factor_cnt + facid <= learnable_params_cnt);
        return &update_g[this->feature_cnt + fid * this->field_cnt * this->factor_cnt
                         + fieldid * this->factor_cnt + facid];
    }
    void ApplyGrad();
    
    AdagradUpdater_Num updater;
    
    ThreadPool *threadpool;
    SpinLock lock_w, lock_v;
};

#endif /* train_ffm_algo_h */
