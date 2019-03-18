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
                   size_t _hidden_layer_size):
    FM_Algo_Abst(_dataPath, _factor_cnt), epoch(_epoch_cnt), hidden_layer_size(_hidden_layer_size) {
        assert(this->feature_cnt != 0);
        threadpool = new ThreadPool(1);
        init();
    }
    
    ~Train_NFM_Algo() {
        delete [] update_g;
        delete threadpool;
        threadpool = NULL;
    }
    
    void init();
    void Train();
    
private:
    Train_NFM_Algo() = delete;
    
    size_t epoch;
    size_t batch_size;
    
    size_t hidden_layer_size;
    Fully_Conn_Layer<Sigmoid> *inputLayer, *outputLayer;
    Sigmoid sigmoid;
    
    size_t learnable_params_cnt;
    
    void batchGradCompute(size_t, size_t);
    void accumWideGrad(size_t, float);
    void accumDeepGrad(size_t, const vector<float>&);
    
    float *update_g;
    inline float* update_W(size_t fid) {
        return &update_g[fid];
    }
    inline float* update_V(size_t fid, size_t facid) {
        return &update_g[this->feature_cnt + fid * this->factor_cnt + facid];
    }
    void ApplyGrad();
    
    float loss;
    size_t accuracy;
    AdagradUpdater_Num updater;
    
    ThreadLocal<Matrix*> tl_fc_input_Matrix, tl_fc_bp_Matrix;
    ThreadLocal<vector<Matrix*> > tl_wrapper;
    
    ThreadPool *threadpool;
};

#endif /* train_nfm_algo_h */
