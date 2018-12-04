//
//  ensembling.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/12/3.
//  Copyright © 2018 SongKuangshi. All rights reserved.
//

#ifndef ensembling_h
#define ensembling_h

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

// Hard majority voting
// Weighted Average Probabilities
class Voting {
public:
    Voting(bool _is_prob_avg_voting) {
        is_prob_avg_voting = _is_prob_avg_voting;
    }
    
    std::shared_ptr<vector<double> > final_result(vector<vector<double> >& sub_results) {
        assert(sub_results.size() > 0 && sub_results[0].size() > 0);
        vector<double> res;
        res.resize(sub_results[0].size());
        
        if (is_prob_avg_voting) {
            for (int i = 0; i < sub_results[0].size(); i++) {
                res[i] = 0;
                for (int j = 0; j < sub_results.size(); j++) {
                    res[i] += sub_results[j][i];
                }
                res[i] /= sub_results.size();
            }
        } else {
            for (int i = 0; i < sub_results.size(); i++) {
                const size_t index = std::distance(sub_results[i].begin(),
                                                   std::max_element(sub_results[i].begin(), sub_results[i].end())
                                                   );
                assert(index < sub_results[0].size());
                res[index]++;
            }
        }
        return std::make_shared<vector<double> >(res);
    }
    
private:
    bool is_prob_avg_voting;
};

// AdaBoost
class AdaBoost {
public:
    explicit AdaBoost(size_t _sample_cnt): sample_cnt(_sample_cnt) {
        weights = new double[_sample_cnt];
        const double init_w = 1.0 / _sample_cnt;
        for (int i = 0; i < _sample_cnt; i++) {
            *(weights + i) = init_w;
        }
    }
    
    ~AdaBoost() {
        delete[] weights;
        _model_weights.clear();
    }
    
    std::shared_ptr<double> ensembling_weak_model(std::vector<bool>& pred_correct_mask) {
        double err_rate = 0.;
        for (int i = 0; i < sample_cnt; i++) {
            if (pred_correct_mask[i] == false)
                err_rate += 1.;
        }
        err_rate /= sample_cnt;
        
        double alpha = model_weighting(err_rate);
        _model_weights.emplace_back(alpha);
        
        double reweighting = std::exp(alpha);
        for (int i = 0; i < sample_cnt; i++) {
            if (pred_correct_mask[i] == false) {
                *(weights + i) *= reweighting;
            } else {
                *(weights + i) /= reweighting;
            }
        }
        return std::make_shared<double>(*weights);
    }
    
    const vector<double>& model_weights() {
        return _model_weights;
    }
    
private:
    inline double model_weighting(double err_rate){
        if (err_rate < 1e-4) {
            return 1000; // strongly outstanding
        }
        // calculate new weight
        return 0.5 * std::log((1 - err_rate) / err_rate);
    }
    
    size_t sample_cnt;
    double* weights = NULL;
    std::vector<double> _model_weights;
};


#endif /* ensembling_h */
