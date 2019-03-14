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
    vector<float> ans;
    int badcase = 0;
    
    vector<float> tmp_vec;
    tmp_vec.resize(fm->factor_cnt);
    
    for (size_t rid = 0; rid < this->test_dataRow_cnt; rid++) { // data row
        float fm_pred = 0.0f;
        if (fm->sumVX != NULL) {
            for (size_t i = 0; i < test_dataSet[rid].size(); i++) { // feature
                const size_t fid = test_dataSet[rid][i].first;
                assert(fid < fm->feature_cnt);
                const float X = test_dataSet[rid][i].second;
                fm_pred += fm->W[fid] * X;
#ifdef FM
                avx_vecScale(fm->getV(fid, 0), tmp_vec.data(), fm->factor_cnt, X);
                fm_pred -= 0.5 * avx_dotProduct(tmp_vec.data(), tmp_vec.data(), fm->factor_cnt);
#endif
            }
#ifdef FM
            fm_pred += 0.5 * avx_dotProduct(fm->getSumVX(rid, 0), fm->getSumVX(rid, 0), fm->factor_cnt);
#endif
        } else {
            // Field-aware FM
            for (size_t i = 0; i < test_dataSet[rid].size(); i++) {
                const size_t fid = test_dataSet[rid][i].first;
                const float X = test_dataSet[rid][i].second;
                const size_t field = test_dataSet[rid][i].field;
                
                fm_pred += fm->W[fid] * X;
                
                for (size_t j = i + 1; j < test_dataSet[rid].size(); j++) {
                    const size_t fid2 = test_dataSet[rid][j].first;
                    const float X2 = test_dataSet[rid][j].second;
                    const size_t field2 = test_dataSet[rid][j].field;
                    
                    float field_w = avx_dotProduct(fm->getV_field(fid, field2, 0),
                                                   fm->getV_field(fid2, field, 0), fm->factor_cnt);
                    fm_pred += field_w * X * X2;
                }
            }
        }
        
        float pCTR = sigmoid.forward(fm_pred);
        if(fm_pred < -30 || fm_pred > 30){
            badcase ++;
        }
        
        ans.emplace_back(pCTR);
    }
    
    if (!test_label.empty()) {
        assert(ans.size() == test_label.size());
        
        float loss = 0;
        int correct = 0;
        for (size_t i = 0; i < test_label.size(); i++) {
            loss += (int)this->test_label[i] == 1 ? -log(ans[i]) : -log(1.0 - ans[i]);
            if (ans[i] > 0.5 && this->test_label[i] == 1) {
                correct++;
            } else if (ans[i] < 0.5 && this->test_label[i] == 0) {
                correct++;
            }
        }
        cout << "total log likelihood = " << loss << " correct = " << setprecision(5) <<
                (float)correct / test_dataRow_cnt << " with badcase = " << badcase;
        
        auc->init(&ans, &test_label);
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
    int nchar, y;
    size_t fid, fieldid;
    float val;
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
        if (with_valid_label) {
            if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
                this->test_label.emplace_back(y);
                pline += nchar + 1;
            }
        }
        if(sscanf(pline, "%zu:%zu:%f%n", &fieldid, &fid, &val, &nchar) >= 2){
            pline += nchar + 1;
            while(pline < line.c_str() + (int)line.length() &&
                  sscanf(pline, "%zu:%zu:%f%n", &fieldid, &fid, &val, &nchar) >= 2){
                pline += nchar + 1;
                if (fid < fm->feature_cnt) {
                    assert(!isnan(fid));
                    assert(!isnan(val));
                    tmp.emplace_back(FMFeature(fid, val, fieldid));
                }
            }
        }
        if (tmp.empty()) {
            continue;
        }
        this->test_dataSet.emplace_back(move(tmp));
    }
    this->test_dataRow_cnt = this->test_dataSet.size();
    assert(test_dataRow_cnt > 0);
}

