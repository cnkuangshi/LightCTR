//
//  dl_algo_abst.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/9.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef dl_algo_abst_h
#define dl_algo_abst_h

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "common/thread_pool.h"
#include "common/barrier.h"
#include "util/loss.h"
#include "train/layer/fullyconnLayer.h"
using namespace std;

enum DL_Algo {DNN, CNN, RNN};

template <typename LossFunction, typename ActivationFunction, typename OutputActivationFunction>
class DL_Algo_Abst {
    
public:
    DL_Algo_Abst(string dataPath, size_t _epoch, size_t _feature_cnt,
                   size_t hidden_size, size_t _multiclass_output_cnt = 1):
    epoch(_epoch), feature_cnt(_feature_cnt),
    multiclass_output_cnt(_multiclass_output_cnt) {
        threadpool = new ThreadPool(thread::hardware_concurrency());
        this->dl_algo = DNN;
        loadDataRow(dataPath);
    }
    virtual ~DL_Algo_Abst() {
        dataSet.clear();
        delete threadpool;
        threadpool = NULL;
        for (size_t i = 0; i < network.size(); i++) {
            delete network[i];
        }
    }
    
    virtual void initNetwork(size_t hidden_size) = 0;
    
    virtual const vector<float>& Predict(size_t, vector<vector<float> >&) = 0;
    virtual void BP(size_t, const vector<Matrix*>&) = 0;
    virtual void applyBP(size_t epoch) const = 0;
    
    void appendNNLayer(Layer_Base* layer) {
        network.push_back(layer);
    }
    
    void Train() {
        
        size_t batch_epoch = 0;
        Barrier barrier;
        for (size_t p = 0; p < epoch; p++) {
            
            GradientUpdater::__global_bTraining = true;
            
            vector<size_t> inner_order(dataRow_cnt);
            iota(inner_order.begin(), inner_order.end(), 0);
            random_shuffle(inner_order.begin(), inner_order.end());
            
            // Mini-Batch SGD and shuffle selected
            barrier.reset(GradientUpdater::__global_minibatch_size);
            size_t i = 0;
            for (; i < dataRow_cnt; i++) {
                const size_t rid = inner_order[i];
                auto task = [&, rid]() {
                    auto pred = Predict(rid, dataSet);
                    
                    assert(pred.size() == multiclass_output_cnt);
                    outputActivFun.forward(pred.data(), pred.size());
                    
                    // init threadLocal var
                    Matrix grad_Matrix(1, multiclass_output_cnt);
                    vector<int> onehot(multiclass_output_cnt);
                    
                    fill(onehot.begin(), onehot.end(), 0);
                    if (multiclass_output_cnt == 1) {
                        onehot[0] = label[rid];
                    } else {
                        onehot[label[rid]] = 1; // label should begin from 0
                    }
                    lossFun.gradient(pred.data(), onehot.data(), grad_Matrix.reference().data(), pred.size());
                    if (multiclass_output_cnt > 1) {
                        // Notice when LossFunction is Logistic annotation next line,
                        // otherwise run this line like square + softmax
                        outputActivFun.backward(grad_Matrix.reference().data(), pred.data(),
                                                grad_Matrix.reference().data(), grad_Matrix.size());
                    }
                    
                    vector<Matrix*> wrapper(1);
                    wrapper[0] = &grad_Matrix;
                    
                    BP(rid, wrapper);
                    
                    barrier.unblock();
                };
                if (dl_algo == RNN) {
                    task(); // force RNN into serialization
                } else {
                    threadpool->addTask(move(task));
                }
                
                if ((i + 1) % GradientUpdater::__global_minibatch_size == 0) {
                    barrier.block();
                    if (unlikely(i + GradientUpdater::__global_minibatch_size >= dataRow_cnt)) {
                        barrier.reset(dataRow_cnt - i - 1);
                    } else {
                        barrier.reset(GradientUpdater::__global_minibatch_size);
                    }
                    
                    applyBP(batch_epoch);
                    validate(batch_epoch++);
                }
            }
            if ((i + 1) % GradientUpdater::__global_minibatch_size != 0) {
                barrier.block();
                applyBP(batch_epoch); // apply residue datarows of minibatch
                validate(batch_epoch++);
            }
        }
        threadpool->wait();
        MemoryPool::Instance().leak_checkpoint();
    }
    
    void validate(size_t batch_epoch) {
        if (batch_epoch % 50 == 0) {
            
            GradientUpdater::__global_bTraining = false;
            
            // Validate Loss
            float loss = 0.0f;
            int correct = 0;
            Barrier barrier(dataRow_cnt);
            for (size_t rid = 0; rid < dataRow_cnt; rid++) {
                auto task = [&, rid]() {
                    auto pred = Predict(rid, dataSet);
                    
                    outputActivFun.forward(pred.data(), pred.size());
                    
                    // init threadLocal var
                    vector<int> onehot(multiclass_output_cnt);
                    onehot.resize(multiclass_output_cnt);
                    
                    auto idx = max_element(pred.begin(), pred.end()) - pred.begin();
                    assert(idx >= 0);
                    if (idx == label[rid]) {
                        correct++;
                    }
                    fill(onehot.begin(), onehot.end(), 0);
                    if (multiclass_output_cnt == 1) {
                        onehot[0] = label[rid];
                    } else {
                        onehot[label[rid]] = 1; // label should begin from 0
                    }
                    loss += lossFun.loss(pred.data(), onehot.data(), pred.size());
                    barrier.unblock();
                };
                if (dl_algo == RNN) {
                    task();
                } else {
                    threadpool->addTask(move(task));
                }
            }
            barrier.block();
            printf("Epoch %zu Loss = %f correct = %.3f\n",
                   batch_epoch, loss, 1.0f * correct / dataRow_cnt);
            
            GradientUpdater::__global_bTraining = true;
        }
    }
    
    virtual void loadDataRow(string dataPath) { // MNIST dataset
        dataSet.clear();
        
        ifstream fin_;
        string line;
        int nchar, y;
        int val, fid = 0;
        fin_.open(dataPath, ios::in);
        if(!fin_.is_open()){
            cout << "open file error!" << endl;
            exit(1);
        }
        
        while(!fin_.eof()){
            vector<float> tmp;
            tmp.resize(feature_cnt);
            getline(fin_, line);
            fill(tmp.begin(), tmp.end(), 0);
            const char *pline = line.c_str();
            if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
                pline += nchar + 1;
                if (multiclass_output_cnt == 1) {
                    y = y < 5 ? 0 : 1;
                } else {
                    assert(y < multiclass_output_cnt);
                }
                label.emplace_back(y);
                fid = 0;
                while(pline < line.c_str() + (int)line.length()
                      && sscanf(pline, "%d%n", &val, &nchar) >= 1){
                    pline += nchar + 1;
                    if (*pline == ',')
                        pline += 1;
                    if (val != 0) {
                        tmp[fid] = val / 255.0;
                    }
                    fid++;
                    if (fid > feature_cnt) {
                        break;
                    }
                }
                dataSet.emplace_back(move(tmp));
                if (dataSet.size() > 500) {
                    break;
                }
            }
        }
        this->dataRow_cnt = this->dataSet.size();
        assert(dataRow_cnt > 0 && label.size() == dataRow_cnt);
    }
    
    void saveModel(size_t epoch) {
        
    }
protected:
    DL_Algo dl_algo;
    
    vector<Layer_Base*> network;
    Layer_Base *inputLayer;
    Layer_Base *outputLayer;
    
    size_t feature_cnt, multiclass_output_cnt, dataRow_cnt;
    
private:
    OutputActivationFunction outputActivFun;
    LossFunction lossFun;
    
    size_t epoch;
    
    vector<vector<float> > dataSet;
    vector<int> label;
    
    ThreadPool *threadpool;
};

#endif /* dl_algo_abst_h */
