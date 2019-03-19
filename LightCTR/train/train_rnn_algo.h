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
        initNetwork(hidden_size);
    }
    Train_RNN_Algo() = delete;
                                  
    ~Train_RNN_Algo() {
    }
    
    void initNetwork(size_t hidden_size) {
        inputLSTM = new LSTM_Unit<ActivationFunction>(28, hidden_size, batch_size);
        this->appendNNLayer(inputLSTM);
        attentionLayer =
            new Attention_Unit<ActivationFunction>(hidden_size, /*fc_hidden*/20, batch_size);
        this->appendNNLayer(attentionLayer);
        fcLayer = new Fully_Conn_Layer<ActivationFunction>(attentionLayer, hidden_size, 72);
        this->appendNNLayer(fcLayer);
        this->outputLayer =
            new Fully_Conn_Layer<ActivationFunction>(fcLayer, 72, this->multiclass_output_cnt);
        this->appendNNLayer(this->outputLayer);
    }
    
    vector<float>& Predict(size_t rid, vector<vector<float> >& dataRow) {
        static Matrix* dataRow_Matrix = new Matrix(1, 28);
        static Matrix* dataRow_Matrix_fc = new Matrix(1, hidden_size, 0);
        static vector<Matrix*> tmp;
        tmp.resize(1);
        tmp[0] = dataRow_Matrix;
        
        auto begin = dataRow[rid].begin();
        auto end = begin;
        FOR(i, batch_size) {
            begin = dataRow[rid].begin() + i * 28;
            end = dataRow[rid].begin() + (i + 1) * 28;
            dataRow_Matrix->pointer()->assign(begin, end);
            inputLSTM->forward(tmp);
        }
        assert(end == dataRow[rid].end());
        
        // Attention Unit
        vector<float> pred = attentionLayer->forward(inputLSTM->seq_output());
        
        assert(pred.size() == hidden_size);
        dataRow_Matrix_fc->loadDataPtr(&pred);
        tmp[0] = dataRow_Matrix_fc;
        return this->fcLayer->forward(tmp);
    }
    
    void BP(size_t rid, const vector<Matrix*>& grad) {
        assert(GradientUpdater::__global_bTraining);
        this->outputLayer->backward(grad);
        inputLSTM->backward(attentionLayer->inputDelta());
    }
    
    void applyBP(size_t epoch) const {
        inputLSTM->applyBatchGradient();
        attentionLayer->applyBatchGradient();
    }
    
private:
    size_t batch_size, hidden_size;
    LSTM_Unit<ActivationFunction>* inputLSTM;
    Attention_Unit<ActivationFunction>* attentionLayer;
    Layer_Base* fcLayer;
};

#endif /* train_rnn_algo_h */
