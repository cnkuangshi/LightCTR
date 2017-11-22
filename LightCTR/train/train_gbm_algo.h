//
//  train_gbm_algo.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/26.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef train_gbm_algo_h
#define train_gbm_algo_h

#include <stdio.h>
#include <cmath>
#include "../common/thread_pool.h"
#include "../util/random.h"
#include "../util/activations.h"
#include "../gbm_algo_abst.h"

class Train_GBM_Algo : public GBM_Algo_Abst {
    struct SplitNodeStat_Thread {
        double sumGrad, sumHess;
        double gain;
        double split_threshold;
        double last_value_toCheck;
        bool dataNAN_go_Right;
        int split_feature_index;
        SplitNodeStat_Thread() {
            gain = 0, split_feature_index = -1, split_threshold = 0;
            dataNAN_go_Right = 0;
            clear();
        }
        inline void clear() {
            sumGrad = 0.0f;
            sumHess = 0.0f;
            last_value_toCheck = 1e-12;
        }
        inline bool needUpdate(double splitGain, size_t split_index) {
            assert(!isnan(splitGain));
            assert(split_index >= 0);
            if (split_feature_index <= split_index) {
                return splitGain > this->gain;
            } else {
                return !(this->gain > splitGain);
            }
        }
    };
public:
    Train_GBM_Algo(string _dataPath, size_t _epoch_cnt, size_t _maxDepth,
                   size_t _minLeafW, size_t _multiclass):
    GBM_Algo_Abst(_dataPath, _maxDepth, _minLeafW, _multiclass), epoch_cnt(_epoch_cnt) {
        proc_cnt = thread::hardware_concurrency();
        init();
        threadpool = new ThreadPool(this->proc_cnt);
    }
    ~Train_GBM_Algo() {
        delete [] sampleDataSetIndex;
        delete [] sampleFeatureSetIndex;
        delete [] dataRow_LocAtTree;
        delete [] splitNodeStat_thread;
    }
    
    void init();
    void Train();
    void flash(RegTreeNode *, size_t);
    void findSplitFeature(size_t, size_t, size_t, bool, size_t);
    void findSplitFeature_Wrapper(size_t, size_t, size_t, size_t);
    
    inline void sample() {
        memset(sampleDataSetIndex, 0, sizeof(bool) * this->dataRow_cnt);
        memset(dataRow_LocAtTree, NULL, sizeof(RegTreeNode*) * this->dataRow_cnt);
        for (size_t rid = 0; rid < this->dataRow_cnt; rid++) {
            if(SampleBinary(0.7))
                sampleDataSetIndex[rid] = 1;
        }
        memset(sampleFeatureSetIndex, 0, sizeof(bool) * this->feature_cnt);
        for (size_t fid = 0; fid < this->feature_cnt; fid++) {
            if(dataSet_feature[fid].size() == 0)
                continue;
            if(SampleBinary(0.7))
                sampleFeatureSetIndex[fid] = 1;
        }
    }
    
    inline double grad(double pred, double label) {
        return pred - label;
    }
    inline double hess(double pred) {
        return pred * (1 - pred);
    }
    inline double weight(double sumGrad, double sumHess) {
        return - ThresholdL1(sumGrad, lambda) / (sumHess + lambda);
    }
    inline double gain(double sumGrad, double sumHess) {
        return ThresholdL1(sumGrad, lambda) * ThresholdL1(sumGrad, lambda) / (sumHess + lambda);
    }
    inline double ThresholdL1(double w, double lambda) {
        if (w > +lambda) return w - lambda;
        if (w < -lambda) return w + lambda;
        return 0.0;
    }
    
private:
    ThreadPool *threadpool;
    mutex lock;
    size_t proc_cnt;
    int proc_left;
    SplitNodeStat_Thread* splitNodeStat_thread;
    
    bool* sampleDataSetIndex;
    bool* sampleFeatureSetIndex;
    RegTreeNode** dataRow_LocAtTree;
    size_t epoch_cnt;
    
    Softmax softmax;
    Sigmoid sigmoid;
    
    double eps_feature_value, lambda, learning_rate;
};

#endif /* train_gbm_algo_h */
