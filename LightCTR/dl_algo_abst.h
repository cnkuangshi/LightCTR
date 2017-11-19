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
#include "common/thread_pool.h"
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
    }
    
    virtual void init(size_t hidden_size) {
        // Multi-Layer Perception
        this->inputLayer = new Fully_Conn_Layer<ActivationFunction>(NULL, feature_cnt, hidden_size);
        this->outputLayer = new Fully_Conn_Layer<ActivationFunction>(inputLayer, hidden_size, multiclass_output_cnt);
    }
    
    virtual vector<double>* Predict(size_t, vector<vector<double> >* const) = 0;
    virtual void BP(size_t, vector<Matrix*>*) = 0;
    virtual void applyBP() const = 0;
    
    void Train() {
        static ThreadLocal<vector<double>* > tl_grad;
        static ThreadLocal<vector<int>* > tl_onehot;
        static ThreadLocal<Matrix*> tl_grad_Matrix;
        static ThreadLocal<vector<Matrix*>* > tl_tmp;
        
        for (size_t p = 0; p < epoch; p++) {
            
            GradientUpdater::__global_bTraining = true;
            threadpool->init();
            
            // Mini-Batch SGD and shuffle selected
            for (size_t rid = 0; rid < dataRow_cnt; rid++) {
                
                auto task = [&, rid]() {
                    vector<double> *pred = Predict(rid, &dataSet);
                    
                    assert(pred->size() == multiclass_output_cnt);
                    outputActivFun.forward(pred);
                    
                    // init threadLocal var
                    vector<double>*& grad = *tl_grad;
                    if (grad == NULL) {
                        grad = new vector<double>();
                        grad->resize(multiclass_output_cnt);
                    }
                    vector<int>*& onehot = *tl_onehot;
                    if (onehot == NULL) {
                        onehot = new vector<int>();
                        onehot->resize(multiclass_output_cnt);
                    }
                    Matrix*& grad_Matrix = *tl_grad_Matrix;
                    if (grad_Matrix == NULL) {
                        grad_Matrix = new Matrix(1, multiclass_output_cnt, 0);
                    }
                    vector<Matrix*>*& tmp = *tl_tmp;
                    if (tmp == NULL) {
                        tmp = new vector<Matrix*>();
                        tmp->resize(1);
                    }
                    
                    fill(onehot->begin(), onehot->end(), 0);
                    if (multiclass_output_cnt == 1) {
                        onehot->at(0) = label[rid];
                    } else {
                        onehot->at(label[rid]) = 1; // label should begin from 0
                    }
                    lossFun.gradient(pred, onehot, grad);
                    if (multiclass_output_cnt > 1) {
                        // Notice when LossFunction is Logistic annotation next line,
                        // otherwise run this line like square + softmax
                        outputActivFun.backward(grad, pred, grad);
                    }
                    grad_Matrix->loadDataPtr(grad);
                    tmp->at(0) = grad_Matrix;
                    
                    BP(rid, tmp);
                };
                if (dl_algo == RNN) {
                    task(); // force RNN into serialization
                } else {
                    threadpool->addTask(task);
                }
                
                if ((rid + 1) % GradientUpdater::__global_minibatch_size == 0) {
                    threadpool->join();
                    applyBP();
                    threadpool->init();
                }
            }
            threadpool->join();
            
            if (p % 2 == 0) {
                
                GradientUpdater::__global_bTraining = false;
                threadpool->init();
                
                // Validate Loss
                std::atomic<double> loss(0.0f);
                std::atomic<int> correct(0);
                for (size_t rid = 0; rid < dataRow_cnt; rid++) {
                    auto task = [&, rid]() {
                        vector<double> *pred = Predict(rid, &dataSet);
                        
                        outputActivFun.forward(pred);
                        
                        // init threadLocal var
                        vector<double>*& grad = *tl_grad;
                        if (grad == NULL) {
                            grad = new vector<double>();
                            grad->resize(multiclass_output_cnt);
                        }
                        vector<int>*& onehot = *tl_onehot;
                        if (onehot == NULL) {
                            onehot = new vector<int>();
                            onehot->resize(multiclass_output_cnt);
                        }
                        Matrix*& grad_Matrix = *tl_grad_Matrix;
                        if (grad_Matrix == NULL) {
                            grad_Matrix = new Matrix(1, multiclass_output_cnt, 0);
                        }
                        vector<Matrix*> *tmp = *tl_tmp;
                        if (tmp == NULL) {
                            tmp = new vector<Matrix*>();
                            tmp->resize(1);
                        }
                        
                        auto idx = max_element(pred->begin(), pred->end()) - pred->begin();
                        assert(idx >= 0);
                        if (idx == label[rid]) {
                            correct++;
                        }
                        fill(onehot->begin(), onehot->end(), 0);
                        if (multiclass_output_cnt == 1) {
                            onehot->at(0) = label[rid];
                        } else {
                            onehot->at(label[rid]) = 1; // label should begin from 0
                        }
                        loss = loss + lossFun.loss(pred, onehot);
                    };
                    if (dl_algo == RNN) {
                        task();
                    } else {
                        threadpool->addTask(task);
                    }
                }
                threadpool->join();
                printf("\nepoch %zu Loss = %lf correct = %.3f\n",
                       p, loss.load(), 1.0f * correct / dataRow_cnt);
            }
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
            vector<double> tmp;
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
                dataSet.emplace_back(tmp);
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
    
    Layer_Base *inputLayer;
    Layer_Base *outputLayer;
    
    size_t feature_cnt, multiclass_output_cnt, dataRow_cnt;
    
private:
    OutputActivationFunction outputActivFun;
    LossFunction lossFun;
    
    size_t epoch;
    
    vector<vector<double> > dataSet;
    vector<int> label;
    
    ThreadPool *threadpool;
};

#endif /* dl_algo_abst_h */
