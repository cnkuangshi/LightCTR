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
    vector<double> ans;
    size_t badcase = 0;
    for (size_t rid = 0; rid < this->test_dataRow_cnt; rid++) { // data row
        double gbm_pred = 0.0f;
        for (size_t tid = 0; tid < gbm->RegTreeRootArr.size(); tid++) {
            gbm_pred += gbm->locAtLeafWeight(gbm->RegTreeRootArr[tid], test_dataSet[rid]);
        }
        if(gbm_pred < -30 || gbm_pred > 30){
            badcase ++;
        }
        assert(!isnan(gbm_pred));
        double pCTR = activFunc(gbm_pred);
        assert(!isnan(pCTR));
        ans.emplace_back(pCTR);
    }
    assert(badcase != test_dataRow_cnt);
    if (!test_label.empty()) {
        assert(ans.size() == test_label.size());
        double loss = 0;
        int correct = 0;
        for (size_t i = 0; i < test_label.size(); i++) {
            assert(ans[i] > 0 && 1.0 - ans[i] > 0);
            loss += (int)this->test_label[i] == 1 ? log(ans[i]) : log(1.0 - ans[i]);
            assert(!isnan(loss));
            if (ans[i] > 0.5 && this->test_label[i] == 1) {
                correct++;
            } else if (ans[i] < 0.5 && this->test_label[i] == 0) {
                correct++;
            }
        }
        cout << "total log likelihood = " << -loss << " correct = " << setprecision(5) <<
                (double)correct / test_dataRow_cnt << " with badcase = " << badcase;
        
        AucEvaluator* auc = new AucEvaluator(&ans, &test_label);
        printf(" auc = %.4f\n", auc->Auc());
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
            y = y < 5 ? 0 : 1;
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

