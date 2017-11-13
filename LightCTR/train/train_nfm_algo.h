//
//  train_nfm_algo.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/6.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef train_nfm_algo_h
#define train_nfm_algo_h

#include <stdio.h>
#include <mutex>
#include "../fm_algo_abst.h"
#include "layer/fullyconnLayer.h"

// Wide-Deep Neural Factorization Machine
class Train_NFM_Algo : public FM_Algo_Abst {
    
public:
    Train_NFM_Algo(string _dataPath, size_t _epoch_cnt, size_t _factor_cnt,
                   size_t _hidden_layer_size, size_t _feature_cnt = 0):
    FM_Algo_Abst(_dataPath, _factor_cnt, _feature_cnt), epoch(_epoch_cnt), hidden_layer_size(_hidden_layer_size) {
        assert(this->feature_cnt != 0);
        threadpool = new ThreadPool(1);
        init();
    }
    
    void init();
    void Train();
    
private:
    size_t epoch;
    size_t batch_size;
    
    size_t hidden_layer_size;
    Fully_Conn_Layer<Sigmoid> *inputLayer, *outputLayer;
    Sigmoid sigmoid;
    
    size_t learnable_params_cnt;
    
    void batchGradCompute(size_t, size_t);
    void accumWideGrad(size_t, double);
    void accumDeepGrad(size_t, vector<double>*);
    
    double *update_g;
    inline double* update_W(size_t fid) {
        return &update_g[fid];
    }
    inline double* update_V(size_t fid, size_t facid) {
        assert(this->feature_cnt + fid * this->factor_cnt + facid < this->feature_cnt * (this->factor_cnt + 1));
        return &update_g[this->feature_cnt + fid * this->factor_cnt + facid];
    }
    void ApplyGrad();
    
    AdagradUpdater_Num updater;
    
    ThreadLocal<Matrix*> tl_fc_input_Matrix, tl_fc_bp_Matrix;
    ThreadLocal<vector<Matrix*>*> tl_wrapper;
    
    ThreadPool *threadpool;
    mutex lock_w, lock_v;
};

#endif /* train_nfm_algo_h */
