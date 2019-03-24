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
#include "../common/avx.h"
using namespace std;

class Activation {
public:
    virtual void forward(float* input, size_t len) = 0;
    virtual void backward(const float* delta, const float* forward_output, float* to, size_t len) = 0;
};

class Identity : public Activation {
public:
    inline void forward(float* input, size_t len) {
        return;
    }
    inline void backward(const float* delta, const float* forward_output, float* to, size_t len) {
        for (size_t i = 0; i < len; i++) {
            to[i] = delta[i];
        }
    }
};

class Binary_Sigmoid : public Activation {
    // used in forward process of Binary Neural Network
public:
    inline float forward(float input) {
        const float res = (input + 1.0f) / 2.0f;
        return fmax(0.0f, fmin(1.0f, res)); // clip to [0, 1]
    }
    inline void forward(float* input, size_t len) {
        float scaler = 0.0f;
        for (size_t i = 0; i < len; i++) {
            scaler += fabs(input[i]); // accumulate of L1-norm
        }
        scaler /= len;
        for (size_t i = 0; i < len; i++) {
            const float sign = input[i] > 0 ? 1 : -1;
            input[i] *= scaler * sign;
        }
    }
    inline void backward(const float* delta, const float* foutput, float* to, size_t len) {
        // standard backward propagation except binary weight
        for (size_t i = 0; i < len; i++) {
            to[i] = delta[i];
        }
    }
};

class Sigmoid : public Activation {
public:
    inline float forward(float input) const {
        if(input < -16){
            return 1e-7;
        } else if(input > 16) {
            return 1.0 - 1e-7;
        }
        return 1.0f / (1.0f + exp(-input));
    }
    inline void forward(float* input, size_t len) {
        for (size_t i = 0; i < len; i++) {
            if(input[i] < -16){
                input[i] = 1e-7;
            } else if(input[i] > 16) {
                input[i] = 1.0 - 1e-7;
            } else {
                input[i] = 1.0f / (1.0f + exp(- input[i]));
            }
            assert(!isnan(input[i]));
        }
    }
    inline void backward(const float* delta, const float* foutput, float* to, size_t len) {
        for (size_t i = 0; i < len; i++) {
            to[i] = delta[i] * foutput[i] * (1.0f - foutput[i]);
            assert(!isnan(to[i]));
        }
    }
};

class Softmax : public Activation {
public:
    Softmax(float _softTargetRate = 1.0f) : softTargetRate(_softTargetRate) {
    }
    inline size_t forward_max(const float* input, size_t len) const {
        return std::max_element(input, input + len) - input;
    }
    inline void forward(float* input, size_t len) {
        float sum = 0.0f;
        auto maxV = *max_element(input, input + len);
        // for numerical stability overflow
        for (size_t i = 0; i < len; i++) {
            sum += exp((input[i] - maxV) / softTargetRate);
        }
        for (size_t i = 0; i < len; i++) {
            input[i] = exp((input[i] - maxV) / softTargetRate) / sum;
            if (input[i] == 0) {
                input[i] = 1e-7;
            } else if (input[i] == 1) {
                input[i] = 1.0 - 1e-7;
            }
        }
    }
    inline void backward(const float* delta, const float* foutput, float* to, size_t len) {
        // softmax Derivative (whether i == j) * softmax(input[i]) - softmax(input[i]) * softmax(input[i])
        // each derivative of Z_(L) = sum_i( delta_(i) * -forward_output_(i) * forward_output_(L) )
        //      + delta_(L) * forward_output_(L)
        float sum = avx_dotProduct(delta, foutput, len);
        avx_vecAdd(delta, -sum, to, len);
        avx_vecScale(to, to, len, foutput);
        avx_vecScale(to, to, len, 1.0 / softTargetRate);
    }
private:
    // used in distillation soft target softmax, when larger than 1 makes smooth classification
    float softTargetRate;
};

class Tanh : public Activation {
public:
    inline void forward(float* input, size_t len) {
        float t1, t2;
        for (size_t i = 0; i < len; i++) {
            t1 = exp(input[i]), t2 = exp(- input[i]);
            input[i] = (t1 - t2) / (t1 + t2);
        }
    }
    inline void backward(const float* delta, const float* foutput, float* to, size_t len) {
        for (size_t i = 0; i < len; i++) {
            to[i] = delta[i] * (1.0f - foutput[i] * foutput[i]);
        }
    }
};

class ReLU : public Activation { // Local Response Normalization
public:
    inline void forward(float* input, size_t len) {
        for (size_t i = 0; i < len; i++) {
            if (input[i] < 0.0f) {
                input[i] = 0.0f; // negative slope is 0
            }
        }
    }
    inline void backward(const float* delta, const float* foutput, float* to, size_t len) {
        for (size_t i = 0; i < len; i++) {
            if (foutput[i] == 0.0f) {
                to[i] = 0.0f;
            } else {
                to[i] = delta[i];
            }
        }
    }
};

class SoftPlus : public Activation {
public:
    inline void forward(float* input, size_t len) {
        for (size_t i = 0; i < len; i++) {
            input[i] = log(1 + exp(input[i]));
        }
    }
    inline void backward(const float* delta, const float* foutput, float* to, size_t len) {
        float t;
        for (size_t i = 0; i < len; i++) {
            t = exp(foutput[i]);
            to[i] = delta[i] * (t - 1) / t;
        }
    }
};

#endif /* activation_h */
