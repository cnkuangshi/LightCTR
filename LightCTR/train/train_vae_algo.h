//
//  train_vae_algo.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/21.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef train_vae_algo_h
#define train_vae_algo_h

#include <stdio.h>
#include <fstream>
#include <iostream>
#include "../util/loss.h"
#include "layer/layer_abst.h"
#include "layer/convLayer.h"
#include "layer/poolingLayer.h"
#include "layer/sampleLayer.h"
using namespace std;

// Generative-Models Variational-Autoencoder
template <typename LossFunction, typename ActivationFunction>
class Train_VAE_Algo {
    
public:
    Train_VAE_Algo(string dataPath, size_t _epoch, size_t _feature_cnt,
                   size_t hidden_size, size_t _gauss_cnt):
    epoch(_epoch), feature_cnt(_feature_cnt), gauss_cnt(_gauss_cnt) {
        loadDenseDataRow(dataPath);
        init(hidden_size, _gauss_cnt);
    }
    ~Train_VAE_Algo() {
        delete encodeLayer;
        delete decodeLayer;
        delete outputLayer;
        delete sampleLayer;
    }
    
    void init(size_t hidden_size, size_t gauss_cnt) {
        // Find expectation nomal distribution
        this->encodeLayer = new Fully_Conn_Layer<ActivationFunction>(NULL, feature_cnt, hidden_size);
        Fully_Conn_Layer<Identity>* hidden1 = new Fully_Conn_Layer<Identity>(encodeLayer, hidden_size, gauss_cnt * 2);
        // sample
        sampleLayer = new Sample_Layer<Identity>(hidden1, gauss_cnt * 2);
        decodeLayer = new Fully_Conn_Layer<ActivationFunction>(sampleLayer, gauss_cnt, hidden_size);
        // tuning parameters to Maximize Likelihood
        this->outputLayer = new Fully_Conn_Layer<ActivationFunction>(decodeLayer, hidden_size, feature_cnt);
    }
    
    void Train() {
        static vector<double> *grad = new vector<double>();
        static Matrix* dataRow_Matrix = new Matrix(1, feature_cnt, 0);
        static Matrix* grad_Matrix = new Matrix(1, feature_cnt, 0);
        static vector<Matrix*> *tmp = new vector<Matrix*>();
        tmp->resize(1);
        
        for (size_t p = 0; p < epoch; p++) {
            
            GradientUpdater::__global_bTraining = true;
            
            // Mini-Batch SGD
            for (size_t rid = 0; rid < dataRow_cnt; rid++) {
                dataRow_Matrix->loadDataPtr(&dataSet[rid]);
                tmp->at(0) = dataRow_Matrix;
                vector<double> *pred = this->encodeLayer->forward(tmp);
                outputActivFun.forward(pred);
                assert(pred->size() == feature_cnt);
                grad->resize(pred->size());
                lossFun.gradient(pred, &dataSet[rid], grad);
                outputActivFun.backward(grad, pred, grad);
                // if LossFunction is Logistic, annotation last line
                grad_Matrix->loadDataPtr(grad);
                tmp->at(0) = grad_Matrix;
                this->outputLayer->backward(tmp);
                if ((rid + 1) % GradientUpdater::__global_minibatch_size == 0) {
                    this->encodeLayer->applyBatchGradient();
                }
            }
            if (p % 2 == 0) {
                
                GradientUpdater::__global_bTraining = false;
                
                // Validate Loss
                double loss = 0.0f;
                for (size_t rid = 0; rid < dataRow_cnt; rid+=2) {
                    dataRow_Matrix->loadDataPtr(&dataSet[rid]);
                    tmp->at(0) = dataRow_Matrix;
                    vector<double> *pred = this->encodeLayer->forward(tmp);
                    outputActivFun.forward(pred);
                    loss += lossFun.loss(pred, &dataSet[rid]);
                    if (rid == 4 || rid == 8) { // look like number 4 or 5
                        for (size_t i = 0; i < feature_cnt; i++) {
                            cout.width(3);
                            cout << int(pred->at(i) * 255) << ",";
                            if ((i + 1) % 28 == 0) {
                                cout << endl;
                            }
                        }
                    }
                }
                printf("\nepoch %zu Loss = %lf\n", p, loss);
            }
        }
    }
    
    vector<double>* encode(vector<double>* input) {
        assert(input->size() == feature_cnt);
        sampleLayer->bEncoding = true;
        vector<double> *encode = this->encodeLayer->forward(input);
        sampleLayer->bEncoding = false;
        assert(encode->size() == gauss_cnt);
        return encode;
    }
    
    vector<double>* decode(vector<double>* input) {
        assert(input->size() == gauss_cnt);
        vector<double> *decode = this->decodeLayer->forward(input);
        outputActivFun.forward(decode);
        return decode;
    }
    
    void loadDenseDataRow(string dataPath) {
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
                fid = 0;
                while(pline < line.c_str() + (int)line.length() &&
                      sscanf(pline, "%d%n", &val, &nchar) >= 1){
                    pline += nchar + 1;
                    if (*pline == ',')
                        pline += 1;
                    if (val != 0) {
                        tmp[fid] = val / 255.0;
                    }
                    fid++;
                    if (fid >= feature_cnt) {
                        break;
                    }
                }
                dataSet.push_back(tmp);
                if (dataSet.size() > 200) {
                    break;
                }
            }
        }
        this->dataRow_cnt = this->dataSet.size();
        assert(this->dataRow_cnt > 0);
        
        for (size_t i = 0; i < feature_cnt; i++) {
            cout.width(3);
            cout << int(dataSet[4][i] * 255) << ",";
            if ((i + 1) % 28 == 0) {
                cout << endl;
            }
        }
        for (size_t i = 0; i < feature_cnt; i++) {
            cout.width(3);
            cout << int(dataSet[8][i] * 255) << ",";
            if ((i + 1) % 28 == 0) {
                cout << endl;
            }
        }
    }
    
private:
    LossFunction lossFun;
    Sigmoid outputActivFun;
    
    size_t epoch;
    Fully_Conn_Layer<ActivationFunction> *encodeLayer, *decodeLayer, *outputLayer;
    Sample_Layer<Identity> *sampleLayer;
    
    size_t dataRow_cnt, feature_cnt, gauss_cnt;
    vector<vector<double> > dataSet;
};

#endif /* train_vae_algo_h */
