//
//  em_algo_abst.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/13.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef em_algo_abst_h
#define em_algo_abst_h

#include <iostream>
#include <cstdio>
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <fstream>
#include "assert.h"
using namespace std;

template <typename T>
class EM_Algo_Abst {
public:
    EM_Algo_Abst(string _dataFile, size_t _epoch, size_t _feature_cnt):
    epoch(_epoch), feature_cnt(_feature_cnt) {
        loadDataRow(_dataFile);
    }
    virtual ~EM_Algo_Abst() {
        dataSet.clear();
    }
    
    void Train() {
        double lastLE = 0;
        for (size_t i = 0; i < this->epoch; i++) {
            T** latentVar = Train_EStep();
            double likelihood = Train_MStep(latentVar);
            assert(!isnan(likelihood));
            cout << "epoch " << i << " log likelihood ELOB = " << likelihood << endl;
            if (i == 0 || fabs(likelihood - lastLE) > 1e-3) {
                lastLE = likelihood;
            } else {
                cout << "have been converge" << endl;
                printArguments();
                break;
            }
        }
    }
    void saveModel(size_t epoch) {
        
    }
    
    virtual void init() = 0;
    virtual T** Train_EStep() = 0;
    virtual double Train_MStep(T**) = 0;
    virtual void printArguments() = 0;
    virtual shared_ptr<vector<int> > Predict() = 0;
    
    void loadDataRow(string dataPath) {
        dataSet.clear();
        
        ifstream fin_;
        string line;
        int nchar;
        double val;
        fin_.open(dataPath, ios::in);
        if(!fin_.is_open()){
            cout << "open file error!" << endl;
            exit(1);
        }
        vector<double> tmp;
        tmp.reserve(feature_cnt);
        while(!fin_.eof()){
            getline(fin_, line);
            tmp.clear();
            const char *pline = line.c_str();
            while(pline < line.c_str() + (int)line.length() &&
                  sscanf(pline, "%lf%n", &val, &nchar) >= 1){
                pline += nchar + 1;
                assert(!isnan(val));
                tmp.emplace_back(val);
                if (tmp.size() == feature_cnt) {
                    assert(tmp.size() == feature_cnt);
                    this->dataSet.emplace_back(move(tmp));
                    tmp.clear();
                }
            }
        }
        this->dataRow_cnt = this->dataSet.size();
        assert(this->dataRow_cnt > 0);
    }
    
    size_t epoch;
    size_t feature_cnt, dataRow_cnt;
    vector<vector<double> > dataSet;
};

#endif /* em_algo_abst_h */
