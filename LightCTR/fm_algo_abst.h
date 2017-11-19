//
//  fm_algo_abst.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/23.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef fm_algo_abst_h
#define fm_algo_abst_h

#include <iostream>
#include <stdio.h>
#include <vector>
#include <set>
#include <thread>
#include <fstream>
#include <string>
#include <cmath>
#include "assert.h"
#include "util/random.h"
#include "util/gradientUpdater.h"
#include "util/momentumUpdater.h"

#define FM

using namespace std;

struct FMFeature {
    size_t first; // feature id
    double second; // value
    size_t field;
    FMFeature(size_t _first, double _second, size_t _field):
    first(_first), second(_second), field(_field) {}
};

class FM_Algo_Abst {
public:
    FM_Algo_Abst(string _dataPath, size_t _factor_cnt,
                 size_t _field_cnt = 0, size_t _feature_cnt = 0):
    feature_cnt(_feature_cnt), field_cnt(_field_cnt), factor_cnt(_factor_cnt), dropout(0.2) {
        proc_cnt = thread::hardware_concurrency();
        loadDataRow(_dataPath);
        init();
    }
    virtual ~FM_Algo_Abst() {
        delete [] W;
#ifdef FM
        delete [] V;
        delete [] sumVX;
#endif
    }
    void init() {
        W = new double[this->feature_cnt];
        memset(W, 0, sizeof(double) * this->feature_cnt);
#ifdef FM
        size_t memsize = this->feature_cnt * this->factor_cnt;
        if (this->field_cnt > 0) {
            memsize = this->feature_cnt * this->field_cnt * this->factor_cnt;
        }
        V = new double[memsize];
        for (size_t i = 0; i < memsize; i++) {
            V[i] = GaussRand() * 0.01f; // 0.01 decay
        }
        sumVX = NULL;
        
        dropout_mask = new bool[this->factor_cnt];
#endif
    }
    
    void loadDataRow(string dataPath) {
        dataSet.clear();
        
        ifstream fin_;
        string line;
        int nchar, y;
        size_t fid, fieldid;
        double val;
        fin_.open(dataPath, ios::in);
        if(!fin_.is_open()){
            cout << "open file error!" << endl;
            exit(1);
        }
        vector<FMFeature> tmp;
        while(!fin_.eof()){
            getline(fin_, line);
            tmp.clear();
            const char *pline = line.c_str();
            if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
                pline += nchar + 1;
                label.emplace_back(y);
                while(pline < line.c_str() + (int)line.length() &&
                      sscanf(pline, "%zu:%zu:%lf%n", &fieldid, &fid, &val, &nchar) >= 2){
                    pline += nchar + 1;
                    tmp.emplace_back(*new FMFeature(fid, val, fieldid));
                    this->feature_cnt = max(this->feature_cnt, fid + 1);
                    if (this->field_cnt > 0) {
                        this->field_cnt = max(this->field_cnt, fieldid + 1);
                    }
                }
            }
            if (tmp.empty()) {
                continue;
            }
            this->dataSet.emplace_back(tmp);
        }
        this->dataRow_cnt = this->dataSet.size();
    }
    
    void saveModel(size_t epoch) {
        char buffer[1024];
        snprintf(buffer, 1024, "%d", (int)epoch);
        string filename = buffer;
        ofstream md("./output/model_epoch_" + filename + ".txt");
        if(!md.is_open()){
            cout<<"save model open file error" << endl;
            exit(1);
        }
        for (size_t fid = 0; fid < this->feature_cnt; fid++) {
            if (W[fid] != 0) {
                md << fid << ":" << W[fid] << " ";
            }
        }
        md << endl;
#ifdef FM
        // print all factor V
        for (size_t fid = 0; fid < this->feature_cnt; fid++) {
            md << fid << ":";
            for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
                md << *getV(fid, fac_itr) << " ";
            }
            md << endl;
        }
#endif
        md.close();
    }
    
    virtual void Train() = 0;
    
    double L2Reg_ratio;
    
    double *W;
    size_t feature_cnt, proc_cnt, field_cnt, factor_cnt;
    size_t dataRow_cnt;
    
    double *V, *sumVX;
    inline double* getV(size_t fid, size_t facid) const {
        assert(fid * this->factor_cnt + facid < this->feature_cnt * this->factor_cnt);
        return &V[fid * this->factor_cnt + facid];
    }
    inline double* getV_field(size_t fid, size_t fieldid, size_t facid) const {
        assert(fid * this->factor_cnt + facid < this->feature_cnt * field_cnt * this->factor_cnt);
        return &V[fid * this->field_cnt * this->factor_cnt + fieldid * this->factor_cnt + facid];
    }
    inline double* getSumVX(size_t rid, size_t facid) const {
        assert(rid * this->factor_cnt + facid < this->dataRow_cnt * this->factor_cnt);
        return &sumVX[rid * this->factor_cnt + facid];
    }
    
    vector<vector<FMFeature> > dataSet;
    
protected:
    DropoutUpdater dropout;
    inline double LogisticGradW(double pred, double label, double x) {
        return (pred - label) * x * dropout.rescale();
    }
    inline double LogisticGradV(double gradW, double sum, double v, double x) {
        return gradW * (sum - v * x);
    }
    
    AdagradUpdater_Num updater;
    
    bool* dropout_mask;
    
    vector<int> label;
    vector<set<int> > cross_field;
};

#endif /* fm_algo_abst_h */
