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
        this->init(_hidden_size);
    }
    ~Train_CNN_Algo() {
        delete this->inputLayer;
        delete this->outputLayer;
    }
    
    void init(size_t hidden_size) {
        // Net structure of 28x28: 5x5 12 pool 6 3x3 4 3x3 2 flatten fc-100
        this->inputLayer = new Conv_Layer<ActivationFunction>(NULL, 1, 6, CNN_Config{5, 0, 2});
        Max_Pooling_Layer<Identity>* poolLayer =
            new Max_Pooling_Layer<Identity>(this->inputLayer, 6, Pool_Config{2});
        Conv_Layer<ActivationFunction>* hidden1 =
            new Conv_Layer<ActivationFunction>(poolLayer, 6, 16, CNN_Config{3, 0, 1});
        Conv_Layer<ActivationFunction>* hidden2 =
            new Conv_Layer<ActivationFunction>(hidden1, 16, 20, CNN_Config{3, 0, 1});
        Adapter_Layer<Identity>* adapter = new Adapter_Layer<Identity>(hidden2, 2);
        Fully_Conn_Layer<ActivationFunction>* fcLayer =
            new Fully_Conn_Layer<ActivationFunction>(adapter, 20 * 2 * 2, hidden_size);
        this->outputLayer =
            new Fully_Conn_Layer<ActivationFunction>(fcLayer, hidden_size,
                                                     this->multiclass_output_cnt);
    }
    
    vector<double>* Predict(size_t rid, vector<vector<double> >* const dataRow) {
        Matrix*& dataRow_Matrix = *tl_dataRow_Matrix;
        if (dataRow_Matrix == NULL) {
            dataRow_Matrix = new Matrix(sqrt((double)this->feature_cnt),
                                        sqrt((double)this->feature_cnt), 0);
        }
        vector<Matrix*>*& wrapper = *tl_wrapper;
        if (wrapper == NULL) {
            wrapper = new vector<Matrix*>();
            wrapper->resize(1);
        }
        dataRow_Matrix->loadDataPtr(&dataRow->at(rid));
        
        wrapper->at(0) = dataRow_Matrix;
        return this->inputLayer->forward(wrapper);
    }
    
    void BP(size_t rid, vector<Matrix*>* grad) {
        assert(GradientUpdater::__global_bTraining);
        this->outputLayer->backward(grad);
    }
    
    void applyBP() const {
        this->inputLayer->applyBatchGradient();
    }
private:
    ThreadLocal<vector<Matrix*>*> tl_wrapper;
    ThreadLocal<Matrix*> tl_dataRow_Matrix;
};

#endif /* train_cnn_algo_h */
