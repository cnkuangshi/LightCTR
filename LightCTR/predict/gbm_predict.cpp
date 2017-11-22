//
//  gbm_predict.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/26.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "gbm_predict.h"
#include <iomanip>

void GBM_Predict::Predict(string savePath) {
    static vector<double> ans, tmp;
    static vector<int> pLabel;
    size_t badcase = 0;
    tmp.resize(gbm->multiclass);
    ans.clear();
    pLabel.clear();
    
    assert(gbm->RegTreeRootArr.size() % gbm->multiclass == 0);
    
    for (size_t rid = 0; rid < this->test_dataRow_cnt; rid++) { // data row
        fill(tmp.begin(), tmp.end(), 0);
        for (size_t tid = 0; tid < gbm->RegTreeRootArr.size(); tid+=gbm->multiclass) {
            for (size_t c = 0; c < gbm->multiclass; c++) {
                tmp[c] += gbm->locAtLeafWeight(gbm->RegTreeRootArr[tid + c],
                                               test_dataSet[rid]);
            }
        }
        for (size_t c = 0; c < gbm->multiclass; c++) {
            assert(!isnan(tmp[c]));
            if(tmp[c] < -30 || tmp[c] > 30){
                badcase ++;
                break;
            }
        }
        double pCTR;
        if (gbm->multiclass == 1) {
            pCTR = sigmoid.forward(tmp[0]);
            pLabel.emplace_back(pCTR > 0.5 ? 1 : 0);
        } else {
            softmax.forward(&tmp);
            size_t idx = softmax.forward_max(&tmp);
            pCTR = tmp[idx];
            pLabel.emplace_back(idx);
        }
        
        assert(!isnan(pCTR));
        ans.emplace_back(pCTR);
    }
    if (badcase == test_dataRow_cnt) {
        puts("Have overfitting");
    }
    
    if (!test_label.empty()) {
        assert(ans.size() == test_label.size());
        double loss = 0;
        int correct = 0;
        for (size_t i = 0; i < test_label.size(); i++) {
            if (gbm->multiclass == 1) {
                assert(ans[i] > 0 && ans[i] < 1);
                loss += (int)this->test_label[i] == 1 ? log(ans[i]) : log(1.0 - ans[i]);
            } else {
                assert(ans[i] > 0 && ans[i] <= 1);
                loss += log(ans[i]);
            }
            
            assert(!isnan(loss));
            if (this->test_label[i] == pLabel[i]) {
                correct++;
            }
        }
        cout << "total log likelihood = " << -loss << " correct = " << setprecision(5) <<
        (double)correct / test_dataRow_cnt << " with badcase = " << badcase;
        
        if (gbm->multiclass == 1) {
            AucEvaluator* auc = new AucEvaluator(&ans, &test_label);
            printf(" auc = %.4f", auc->Auc());
        }
        printf("\n");
    }
}

void GBM_Predict::loadDataRow(string dataPath, bool with_valid_label) {
    test_dataSet.clear();
    test_label.clear();
    
    ifstream fin_;
    string line;
    int nchar, y;
    size_t fid, rid = 0;
    int val;
    fin_.open(dataPath, ios::in);
    if(!fin_.is_open()){
        cout << "open file error!" << endl;
        exit(1);
    }
    map<size_t, double> tmp;
    while(!fin_.eof()){
        getline(fin_, line);
        tmp.clear();
        const char *pline = line.c_str();
        if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
            pline += nchar + 1;
            if (gbm->multiclass > 1) {
                assert(y < gbm->multiclass);
            } else {
                y = y < 5 ? 0 : 1;
            }
            test_label.emplace_back(y);
            fid = 0;
            while(pline < line.c_str() + (int)line.length() &&
                  sscanf(pline, "%d%n", &val, &nchar) >= 1){
                pline += nchar + 1;
                if (*pline == ',')
                    pline += 1;
                fid++;
                if (val == 0) {
                    continue;
                }
                tmp[fid] = val;
            }
            assert(!tmp.empty());
        }
        if (tmp.empty()) {
            continue;
        }
        this->test_dataSet.emplace_back(tmp);
        rid++;
    }
    this->test_dataRow_cnt = this->test_dataSet.size();
    assert(test_dataRow_cnt > 0 && test_label.size() == test_dataRow_cnt);
}

