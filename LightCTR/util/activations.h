//
//  activation.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/20.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef activation_h
#define activation_h

#include <algorithm>
#include <cmath>
#include <vector>
#include "assert.h"
using namespace std;

class Activation {
public:
    virtual inline void forward(vector<double>* input) = 0;
    virtual inline void backward(const vector<double>* delta, const vector<double>* forward_output, vector<double>* to) = 0;
};

class Identity : public Activation {
public:
    inline void forward(vector<double>* input) {
        return;
    }
    inline void backward(const vector<double>* delta, const vector<double>* foutput, vector<double>* to) {
        assert(delta->size() == foutput->size());
        assert(to->size() == foutput->size());
        for (size_t i = 0; i < delta->size(); i++) {
            to->at(i) = delta->at(i);
        }
    }
};

class Binary_Sigmoid : public Activation {
    // used in forward process of Binary Neural Network
public:
    inline double forward(double input) {
        const double res = (input + 1.0f) / 2.0f;
        return fmax(0.0f, fmin(1.0f, res)); // clip to [0, 1]
    }
    inline void forward(vector<double>* input) {
        double scaler = 0.0f;
        for (auto it = input->begin(); it != input->end(); it++) {
            scaler += abs(*it); // accumulate of L1-norm
        }
        scaler /= input->size();
        for (auto it = input->begin(); it != input->end(); it++) {
            double sign = *it > 0 ? 1 : -1;
            *it = *it * scaler * sign;
        }
    }
    inline void backward(const vector<double>* delta, const vector<double>* foutput, vector<double>* to) {
        // standard backward propagation except binary weight
        assert(delta->size() == foutput->size());
        assert(to->size() == foutput->size());
        for (size_t i = 0; i < delta->size(); i++) {
            to->at(i) = delta->at(i);
        }
    }
};

class Sigmoid : public Activation {
public:
    inline double forward(double input) {
        if(input < -30){
            return 1e-12;
        } else if(input > 30) {
            return 1.0 - 1e-12;
        }
        return 1.0f / (1.0f + exp(-input));
    }
    inline void forward(vector<double>* input) {
        for (auto it = input->begin(); it != input->end(); it++) {
            assert(!isnan(*it));
            if(*it < -30){
                *it = 1e-12;
            } else if(*it > 30) {
                *it = 1.0 - 1e-12;
            } else {
                *it = 1.0f / (1.0f + exp(- (*it)));
            }
            assert(!isnan(*it));
        }
    }
    inline void backward(const vector<double>* delta, const vector<double>* foutput, vector<double>* to) {
        assert(delta->size() == foutput->size());
        assert(to->size() == foutput->size());
        for (size_t i = 0; i < delta->size(); i++) {
            to->at(i) = delta->at(i) * foutput->at(i) * (1.0f - foutput->at(i));
            assert(!isnan(to->at(i)));
        }
    }
};

class Softmax : public Activation {
public:
    Softmax(double _softTargetRate = 1.0f) : softTargetRate(_softTargetRate) {
    }
    inline size_t forward_max(vector<double>* input) {
        return max_element(input->begin(), input->end()) - input->begin();
    }
    inline void forward(vector<double>* input) {
        double sum = 0.0f;
        auto maxV = *max_element(input->begin(), input->end());
        // for numerical stability overflow
        for (auto it = input->begin(); it != input->end(); it++) {
            sum += exp((*it - maxV) / softTargetRate);
        }
        for (auto it = input->begin(); it != input->end(); it++) {
            *it = exp((*it - maxV) / softTargetRate) / sum;
            if (*it == 0) {
                *it = 1e-12;
            } else if (*it == 1) {
                *it = 1.0 - 1e-12;
            }
        }
    }
    inline void backward(const vector<double>* delta, const vector<double>* foutput, vector<double>* to) {
        assert(delta->size() == foutput->size());
        assert(to->size() == foutput->size());
        // softmax Derivative (whether i == j) * softmax(input[i]) - softmax(input[i]) * softmax(input[i])
        // each derivative of Z_(L) = sum_i( delta_(i) * -forward_output_(i) * forward_output_(L) )
        //      + delta_(L) * forward_output_(L)
        double sum = 0.0f;
        for (size_t i = 0; i < delta->size(); i++) {
            sum += delta->at(i) * foutput->at(i);
        }
        for (size_t i = 0; i < delta->size(); i++) {
            to->at(i) = (delta->at(i) - sum) * foutput->at(i);
            to->at(i) /= softTargetRate;
        }
    }
private:
    // used in distillation soft target softmax, when larger than 1 makes smooth classification
    double softTargetRate;
};

class Tanh : public Activation {
public:
    inline void forward(vector<double>* input) {
        double t1, t2;
        for (auto it = input->begin(); it != input->end(); it++) {
            t1 = exp(*it), t2 = exp(- (*it));
            *it = (t1 - t2) / (t1 + t2);
        }
    }
    inline void backward(const vector<double>* delta, const vector<double>* foutput, vector<double>* to) {
        assert(delta->size() == foutput->size());
        assert(to->size() == foutput->size());
        for (size_t i = 0; i < delta->size(); i++) {
            to->at(i) = delta->at(i) * (1.0f - foutput->at(i) * foutput->at(i));
        }
    }
};

class ReLU : public Activation { // Local Response Normalization
public:
    inline void forward(vector<double>* input) {
        for (auto it = input->begin(); it != input->end(); it++) {
            if (*it < 0.0f) {
                *it = 0.0f; // negative slope is 0
            }
        }
    }
    inline void backward(const vector<double>* delta, const vector<double>* foutput, vector<double>* to) {
        assert(delta->size() == foutput->size());
        assert(to->size() == foutput->size());
        for (size_t i = 0; i < delta->size(); i++) {
            if (foutput->at(i) == 0.0f) {
                to->at(i) = 0.0f;
            } else {
                to->at(i) = delta->at(i);
            }
        }
    }
};

class SoftPlus : public Activation {
public:
    inline void forward(vector<double>* input) {
        for (auto it = input->begin(); it != input->end(); it++) {
            *it = log(1 + exp(*it));
        }
    }
    inline void backward(const vector<double>* delta, const vector<double>* foutput, vector<double>* to) {
        assert(delta->size() == foutput->size());
        assert(to->size() == foutput->size());
        double t;
        for (size_t i = 0; i < delta->size(); i++) {
            t = exp(foutput->at(i));
            to->at(i) = delta->at(i) * (t - 1) / t;
        }
    }
};

#endif /* activation_h */
