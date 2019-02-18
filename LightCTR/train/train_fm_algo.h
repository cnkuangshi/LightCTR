//
//  train_fm_algo.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/23.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef train_fm_algo_h
#define train_fm_algo_h

#include "../fm_algo_abst.h"
#include <mutex>
#include <cmath>
#include "../util/activations.h"
#include "../util/gradientUpdater.h"
#include "../common/thread_pool.h"
#include "../common/lock.h"
using namespace std;

class Train_FM_Algo : public FM_Algo_Abst {
public:
    Train_FM_Algo(string _dataPath, size_t _epoch_cnt,
                  size_t _factor_cnt):
    FM_Algo_Abst(_dataPath, _factor_cnt), epoch_cnt(_epoch_cnt) {
        assert(this->feature_cnt != 0);
        init();
        threadpool = new ThreadPool(this->proc_cnt);
    }
    Train_FM_Algo() = delete;
    
    ~Train_FM_Algo() {
        delete [] update_g;
        delete threadpool;
        threadpool = NULL;
    }
    
    void init();
    void Train();
    
private:
    ThreadPool *threadpool;
    SpinLock lock;
    int proc_data_left;
    size_t epoch_cnt;
    
    size_t learnable_params_cnt;
    
    void flash();
    
    Sigmoid sigmoid;
    vector<float>* *update_threadLocal;
    
    void batchGradCompute(size_t, size_t, size_t);
    void accumWVGrad(size_t, float, vector<float>*);

    float *update_g;
    inline float* update_W(size_t fid) const {
        return &update_g[fid];
    }
    inline float* update_V(size_t fid, size_t facid) const {
        assert(this->feature_cnt + fid * this->factor_cnt + facid < this->feature_cnt * (this->factor_cnt + 1));
        return &update_g[this->feature_cnt + fid * this->factor_cnt + facid];
    }
    void ApplyGrad();
};

#endif /* train_fm_algo_h */
