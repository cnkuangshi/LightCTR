//
//  distributed_algo_abst.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/17.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef distributed_algo_abst_h
#define distributed_algo_abst_h

#include <unordered_map>
#include <string>
#include <cmath>
#include "distribut/pull.h"
#include "distribut/push.h"
#include "util/random.h"
#include "util/activations.h"
#include "common/thread_pool.h"

#include "fm_algo_abst.h"
using namespace std;

typedef size_t Key;
struct Value {
    float w;
    Value() {
        w = 0.0f;
    }
    Value(const Value &x) {
        w = x.w;
    }
    Value(float _w): w(_w) {
    }
    Value& operator+ (const Value &x) {
        w += x.w;
        return *this;
    }
    Value& operator- (const Value &x) {
        w -= x.w;
        return *this;
    }
    Value& operator* (const Value &x) {
        w *= x.w;
        return *this;
    }
    Value& operator* (float x) {
        w *= x;
        return *this;
    }
    Value& operator/ (const Value &x) {
        w /= x.w;
        return *this;
    }
    Value& operator/ (float x) {
        w /= x;
        return *this;
    }
    Value& operator= (const Value &x) {
        w = x.w;
        return *this;
    }
    Value& operator=(Value &&x) {
        w = x.w;
        return *this;
    }
    void initParam() {
        w = 0.0f; // init by zero or GaussRand()
    }
    bool checkValid() const {
        return !isnan(w) && !isinf(w);
    }
    bool checkPreferredValue() const {
        // ignore obsolete feature
        return abs(w) > 1e-12 && abs(w) < 15;
    }
    void sqrt(Value& newValue) const { // return new instance
        assert(w >= 0);
        newValue.w = ::sqrt(w + 1e-12);
    }
    string toString() const {
        stringstream ss;
        ss << w;
        return ss.str();
    }
    Value(Value &&) = delete;
//    Value &operator=(Value &&) = delete;
};

class Distributed_Algo_Abst {
public:
    Distributed_Algo_Abst(string _dataPath, size_t _epoch_cnt):
    epoch(_epoch_cnt) {
        size_t cur_node_id = worker.Rank();
        stringstream ss;
        ss << _dataPath << "_" << cur_node_id << ".csv";
        loadDataRow(ss.str());
        
        L2Reg_ratio = 0.f;
        batch_size = GradientUpdater::__global_minibatch_size;
    }
    
    ~Distributed_Algo_Abst() {
    }
    
    void Train() {
        GradientUpdater::__global_bTraining = true;
        
        vector<float> loss_curve, accuracy_curve;
        for (size_t i = 0; i < this->epoch; i++) {
            train_loss = 0;
            accuracy = 0;
            
            size_t minibatch_epoch = (this->dataRow_cnt + this->batch_size - 1) / this->batch_size;
            
            for (size_t p = 0; p < minibatch_epoch; p++) {
                size_t start_pos = p * batch_size;
                
                batchGradCompute(i + 1, start_pos,
                                 min(start_pos + batch_size, this->dataRow_cnt),
                                 false);
            }
            
            printf("[Worker Train] epoch = %zu loss = %f\n", i, -train_loss);
            loss_curve.push_back(train_loss);
            accuracy_curve.push_back(1.0 * accuracy / dataRow_cnt);
        }
        
        for (int i = 0; i < this->epoch; i++) {
            printf("%f(%.3f) ", loss_curve[i], accuracy_curve[i]);
        }
        puts("");
        
        puts("Train Task Complete");
        GradientUpdater::__global_bTraining = false;
        
        worker.shutdown([this]() {
            terminate_barrier.unblock();
        });
        terminate_barrier.block();
    }
    
    // Async-SGD
    void batchGradCompute(size_t epoch, size_t rbegin, size_t rend, bool predicting) {
        // TODO cache replacement and transmission algorithms
        pull_map.clear();
        push_map.clear();
        
        for (size_t rid = rbegin; rid < rend; rid++) { // data row
            for (size_t i = 0; i < dataSet[rid].size(); i++) {
                const size_t fid = dataSet[rid][i].first;
                assert(fid < this->feature_cnt);
                
                if (pull_map.count(fid) == 0) { // keys need unique
                    // obsolete feature will be default 0
                    pull_map.insert(make_pair(fid, Value()));
                }
            }
        }
        
        // Pull lastest batch parameters from PS
        if (pull_map.size() > 0) {
            size_t res = 0;
            do {
                res = worker.pull_op.sync(pull_map);
                if (res != pull_map.size()) { // wait for other workers
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            } while(res != pull_map.size());
        }
        
        for (size_t rid = rbegin; rid < rend; rid++) { // data row
            float pred = 0.0f;
            for (size_t i = 0; i < dataSet[rid].size(); i++) {
                if (dataSet[rid][i].second == 0) {
                    continue;
                }
                const size_t fid = dataSet[rid][i].first;
                const Value param = pull_map[fid];
                
                const float X = dataSet[rid][i].second;
                pred += param.w * X;
            }
            
            float pCTR = sigmoid.forward(pred);
            train_loss += (int)this->label[rid] == 1 ?
                                    -log(pCTR) : -log(1.0 - pCTR);
            assert(!isnan(train_loss));
            if (pCTR >= 0.5 && this->label[rid] == 1) {
                accuracy++;
            } else if (pCTR < 0.5 && this->label[rid] == 0) {
                accuracy++;
            }
            
            if (predicting) {
                continue;
            }
            
            const float loss = pCTR - label[rid];
            
            for (size_t i = 0; i < dataSet[rid].size(); i++) {
                if (dataSet[rid][i].second == 0) {
                    continue;
                }
                const size_t fid = dataSet[rid][i].first;
                const Value param = pull_map[fid];
                const float X = dataSet[rid][i].second;
                
                const float gradW = loss * X + L2Reg_ratio * param.w;
                assert(gradW < 100);
                
                auto it = push_map.find(fid);
                if (it == push_map.end()) {
                    push_map.insert(make_pair(fid, Value(gradW)));
                } else {
                    Value grad(gradW);
                    it->second + grad;
                    assert(it->second.checkValid());
                }
            }
        }
        
        // Push grads to PS
        if (push_map.size() > 0)
            worker.push_op.sync(push_map, epoch);
    }
    
private:
    void loadDataRow(string dataPath) {
        dataSet.clear();
        
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
            if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
                pline += nchar + 1;
                label.emplace_back(y);
                while(pline < line.c_str() + (int)line.length() &&
                      sscanf(pline, "%zu:%zu:%f%n", &fieldid, &fid, &val, &nchar) >= 2){
                    pline += nchar + 1;
                    tmp.emplace_back(FMFeature(fid, val, fieldid));
                    this->feature_cnt = max(this->feature_cnt, fid + 1);
                }
            }
            if (tmp.empty()) {
                continue;
            }
            this->dataSet.emplace_back(tmp);
        }
        this->dataRow_cnt = this->dataSet.size();
    }
    
    vector<vector<FMFeature> > dataSet;
    vector<int> label;
    size_t feature_cnt{0};
    size_t dataRow_cnt{0};
    
    float L2Reg_ratio;
    
    Barrier terminate_barrier;
    
    unordered_map<Key, Value> pull_map;
    unordered_map<Key, Value> push_map;
    
    Worker<Key, Value> worker;
    
    float train_loss = 0;
    size_t accuracy{0};
    Sigmoid sigmoid;
    
    size_t epoch;
    size_t batch_size;
};

#endif /* distributed_algo_abst_h */
