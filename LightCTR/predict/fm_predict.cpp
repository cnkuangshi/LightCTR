//
//  fm_predict.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/24.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "fm_predict.h"
#include <iomanip>

void FM_Predict::Predict(string savePath) {
    vector<double> ans;
    int badcase = 0;
    for (size_t rid = 0; rid < this->test_dataRow_cnt; rid++) { // data row
        double fm_pred = 1.0f;
        
        for (size_t i = 0; i < test_dataSet[rid].size(); i++) { // feature
            size_t fid = test_dataSet[rid][i].first;
            assert(fid < fm->feature_cnt);
            double X = test_dataSet[rid][i].second;
            fm_pred += fm->W[fid] * X;
#ifdef FM
            for (size_t fac_itr = 0; fac_itr < fm->factor_cnt; fac_itr++) {
                double tmp = *fm->getV(fid, fac_itr) * X;
                fm_pred -= 0.5 * tmp * tmp;
            }
#endif
        }
#ifdef FM
        for (size_t fac_itr = 0; fac_itr < fm->factor_cnt; fac_itr++) {
            fm_pred += 0.5 * (*fm->getSumVX(rid, fac_itr)) * (*fm->getSumVX(rid, fac_itr));
        }
#endif
        double pCTR = activFunc(fm_pred);
        if(fm_pred < -30 || fm_pred > 30){
            badcase ++;
        }
        
        ans.push_back(pCTR);
    }
    
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
    if (savePath != "") {
        ofstream md(savePath);
        if(!md.is_open()){
            cout << "save model open file error" << endl;
            exit(0);
        }
        for (auto val : ans) {
            md << val << endl;
        }
        md.close();
    }
}

void FM_Predict::loadDataRow(string dataPath, bool with_valid_label) {
    test_dataSet.clear();
    test_label.clear();
    
    ifstream fin_;
    string line;
    int nchar, y, fid, fieldid;
    double val;
    fin_.open(dataPath, ios::in);
    if(!fin_.is_open()){
        cout << "open file error!" << endl;
        exit(1);
    }
    vector<pair<int, double> > tmp;
    while(!fin_.eof()){
        getline(fin_, line);
        tmp.clear();
        const char *pline = line.c_str();
        if (with_valid_label) {
            if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
                this->test_label.push_back(y);
                pline += nchar + 1;
            }
        }
        if(sscanf(pline, "%d:%d:%lf%n", &fieldid, &fid, &val, &nchar) >= 2){
            pline += nchar + 1;
            while(pline < line.c_str() + (int)line.length() &&
                  sscanf(pline, "%d:%d:%lf%n", &fieldid, &fid, &val, &nchar) >= 2){
                pline += nchar + 1;
                if (fid < fm->feature_cnt) {
                    assert(!isnan(fid));
                    assert(!isnan(val));
                    tmp.push_back(make_pair(fid, val));
                }
            }
        }
        if (tmp.empty()) {
            continue;
        }
        this->test_dataSet.push_back(tmp);
    }
    this->test_dataRow_cnt = this->test_dataSet.size();
    assert(test_dataRow_cnt > 0);
}

