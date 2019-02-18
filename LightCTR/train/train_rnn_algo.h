//
//  train_rnn_algo.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/9.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef train_rnn_algo_h
#define train_rnn_algo_h

#include "../dl_algo_abst.h"
#include "unit/lstm_unit.h"
#include "unit/attention_unit.h"
using namespace std;

template <typename LossFunction, typename ActivationFunction, typename OutputActivationFunction>
class Train_RNN_Algo : public DL_Algo_Abst<LossFunction, ActivationFunction,
                              OutputActivationFunction> {
public:
    Train_RNN_Algo(string dataPath, size_t _epoch, size_t _feature_cnt,
                   size_t _hidden_size, size_t _recurrent_cnt, size_t _multiclass_output_cnt = 1):
    DL_Algo_Abst<LossFunction, ActivationFunction, OutputActivationFunction>(
                   dataPath, _epoch, _feature_cnt, _hidden_size, _multiclass_output_cnt),
    batch_size(_recurrent_cnt), hidden_size(_hidden_size) {
        this->dl_algo = RNN;
        this->init(_hidden_size);
    }
    Train_RNN_Algo() = delete;
                                  
    ~Train_RNN_Algo() {
        delete this->inputLayer;
        delete this->attentionLayer;
        delete this->fcLayer;
        delete this->outputLayer;
    }
    
    void init(size_t hidden_size) {
        inputLayer = new LSTM_Unit<ActivationFunction>(28, hidden_size, batch_size);
        attentionLayer =
            new Attention_Unit<ActivationFunction>(hidden_size, /*fc_hidden*/20, batch_size);
        fcLayer = new Fully_Conn_Layer<ActivationFunction>(attentionLayer, hidden_size, 72);
        outputLayer =
            new Fully_Conn_Layer<ActivationFunction>(fcLayer, 72, this->multiclass_output_cnt);
    }
    
    vector<float>* Predict(size_t rid, vector<vector<float> >* const dataRow) {
        static Matrix* dataRow_Matrix = new Matrix(1, 28);
        static Matrix* dataRow_Matrix_fc = new Matrix(1, hidden_size, 0);
        static vector<Matrix*> *tmp = new vector<Matrix*>();
        tmp->resize(1);
        
        vector<float> *pred = NULL;
        tmp->at(0) = dataRow_Matrix;
        
        auto begin = dataRow->at(rid).begin();
        auto end = begin;
        FOR(i, batch_size) {
            begin = dataRow->at(rid).begin() + i * 28;
            end = dataRow->at(rid).begin() + (i + 1) * 28;
            dataRow_Matrix->pointer()->assign(begin, end);
            pred = this->inputLayer->forward(tmp);
        }
        assert(end == dataRow->at(rid).end());
        
        // Attention Unit
        pred = attentionLayer->forward(this->inputLayer->seq_output());
        
        assert(pred && pred->size() == hidden_size);
        dataRow_Matrix_fc->loadDataPtr(pred);
        tmp->at(0) = dataRow_Matrix_fc;
        return this->fcLayer->forward(tmp);
    }
    
    void BP(size_t rid, vector<Matrix*>* grad) {
        assert(GradientUpdater::__global_bTraining);
        this->outputLayer->backward(grad);
        this->inputLayer->backward(this->attentionLayer->inputDelta());
    }
    
    void applyBP(size_t epoch) const {
        this->inputLayer->applyBatchGradient();
        this->attentionLayer->applyBatchGradient();
    }
    
private:
    size_t batch_size, hidden_size;
    
    LSTM_Unit<ActivationFunction> *inputLayer;
    Attention_Unit<ActivationFunction> *attentionLayer;
    Fully_Conn_Layer<ActivationFunction>* fcLayer;
    Fully_Conn_Layer<ActivationFunction> *outputLayer;
};

#endif /* train_rnn_algo_h */
