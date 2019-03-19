//
//  train_cnn_algo.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/9.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef train_cnn_algo_h
#define train_cnn_algo_h

#include "../dl_algo_abst.h"
#include "layer/poolingLayer.h"
#include "layer/adapterLayer.h"
#include "layer/convLayer.h"
#include "../distribut/ring_collect.h"
using namespace std;

template <typename LossFunction, typename ActivationFunction, typename OutputActivationFunction>
class Train_CNN_Algo : public DL_Algo_Abst<LossFunction,
                              ActivationFunction, OutputActivationFunction> {
public:
    Train_CNN_Algo(string dataPath, size_t _epoch, size_t _feature_cnt,
                   size_t _hidden_size, size_t _multiclass_output_cnt = 1):
    DL_Algo_Abst<LossFunction, ActivationFunction, OutputActivationFunction>(
                   dataPath, _epoch, _feature_cnt, _hidden_size, _multiclass_output_cnt) {
        this->dl_algo = CNN;
        initNetwork(_hidden_size);
    }
    Train_CNN_Algo() = delete;
    ~Train_CNN_Algo() {
#ifdef WORKER_RING
        delete syncer;
#endif
    }
    
    void initNetwork(size_t hidden_size) {
        // Net structure of 28x28: 5x5 12 pool 6 3x3 4 3x3 2 flatten fc-100
        this->inputLayer = new Conv_Layer<ActivationFunction>(NULL, 1, 6, CNN_Config{5, 0, 2});
        this->appendNNLayer(this->inputLayer);
        
        Layer_Base* poolLayer =
            new Max_Pooling_Layer<Identity>(this->inputLayer, 6, Pool_Config{2});
        this->appendNNLayer(poolLayer);
        
        Layer_Base* hidden1 =
            new Conv_Layer<ActivationFunction>(poolLayer, 6, 16, CNN_Config{3, 0, 1});
        this->appendNNLayer(hidden1);
        
        Layer_Base* hidden2 =
            new Conv_Layer<ActivationFunction>(hidden1, 16, 20, CNN_Config{3, 0, 1});
        this->appendNNLayer(hidden2);
        
        Layer_Base* adapter = new Adapter_Layer<Identity>(hidden2, 2);
        this->appendNNLayer(adapter);
        
        Layer_Base* fcLayer =
            new Fully_Conn_Layer<ActivationFunction>(adapter, 20 * 2 * 2, hidden_size);
        this->appendNNLayer(fcLayer);
        
        this->outputLayer = new Fully_Conn_Layer<ActivationFunction>(fcLayer, hidden_size,
                                                     this->multiclass_output_cnt);
        this->appendNNLayer(this->outputLayer);
#ifdef WORKER_RING
        syncer = new Worker_RingReduce<float>(__global_cluster_worker_cnt);
        auto buf_fusion = std::make_shared<BufferFusion<float> >(false, false);
        this->inputLayer->registerInitializer(buf_fusion);
        syncer->syncInitializer(buf_fusion);
        puts("[RING] Sync initializer complete");
#endif
    }
    
    const vector<float>& Predict(size_t rid, vector<vector<float> >& dataRow) {
        Matrix*& dataRow_Matrix = *tl_dataRow_Matrix;
        if (dataRow_Matrix == NULL) {
            dataRow_Matrix = new Matrix(sqrt((float)this->feature_cnt),
                                        sqrt((float)this->feature_cnt), 0);
        }
        dataRow_Matrix->loadDataPtr(&dataRow[rid]);
        
        vector<Matrix*> wrapper;
        wrapper.resize(1);
        wrapper[0] = dataRow_Matrix;
        return this->inputLayer->forward(wrapper);
    }
    
    void BP(size_t rid, const vector<Matrix*>& grad) {
        this->outputLayer->backward(grad);
    }
    
    void applyBP(size_t epoch) const {
#ifdef WORKER_RING
        auto buf_fusion = std::make_shared<BufferFusion<float> >(false, false);
        this->inputLayer->registerGradient(buf_fusion);
        syncer->syncGradient(buf_fusion, epoch);
#endif
        this->inputLayer->applyBatchGradient();
    }
private:
    Worker_RingReduce<float>* syncer;
    ThreadLocal<Matrix*> tl_dataRow_Matrix;
};

#endif /* train_cnn_algo_h */
