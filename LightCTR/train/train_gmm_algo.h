//
//  train_gmm_algo.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/13.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef train_gmm_algo_h
#define train_gmm_algo_h

#include <stdio.h>
#include <string>
#include "../common/thread_pool.h"
#include "../em_algo_abst.h"
using namespace std;

class Train_GMM_Algo : public EM_Algo_Abst<vector<float> > {

    struct Point {
        float* data;
        ~Point() {
            delete [] data;
        }
    };
    struct Gauss {
        float* mu;
        float* sigma; // simple covariance to diagonal matrix
        float weight;
        float pdf_tmp;
        float sumRid_tmp;
        Gauss() {
            pdf_tmp = sumRid_tmp = 0;
        }
    };
public:
    Train_GMM_Algo(string _dataFile, size_t _epoch, size_t _cluster_cnt,
                   size_t _feature_cnt, float _scale = 1.0f):
    EM_Algo_Abst(_dataFile, _epoch, _feature_cnt), cluster_cnt(_cluster_cnt), scale(_scale) {
        threadpool = new ThreadPool(thread::hardware_concurrency());
        init();
    }
    Train_GMM_Algo() = delete;
    
    ~Train_GMM_Algo() {
        for (size_t i = 0; i < cluster_cnt; i++) {
            delete [] gaussModels[i].mu;
        }
        delete [] gaussModels;
        delete threadpool;
        threadpool = NULL;
    }

    void init();
    vector<float>* Train_EStep();
    float Train_MStep(const vector<float>*);
    shared_ptr<vector<int> > Predict();
    
    float GaussianLPDF(size_t gasid, size_t rid);
    void printArguments();
    
    size_t cluster_cnt;
    
private:
    float scale;
    Gauss *gaussModels;
    vector<float> latentVar;
    
    ThreadPool *threadpool;
};

#endif /* train_gmm_algo_h */
