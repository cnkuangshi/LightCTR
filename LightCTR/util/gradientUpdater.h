//
//  gradientUpdater.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/23.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef gradientUpdater_h
#define gradientUpdater_h

#include <string.h>
#include "matrix.h"
#include "../common/avx.h"

class GradientUpdater {
public:
    // update weight with L2 Regularization
    inline static void update(float* weight, float grad) {
        *weight += grad + __global_lambdaL2 * *weight;
    }
    inline static void updateL1(float* weight, float grad) {
        *weight += grad + ThresholdL1(*weight);
    }
    inline static void update(vector<float>::iterator weight, float grad) {
        *weight += grad + __global_lambdaL2 * *weight;
    }
    inline static void decay(float ratio) {
        __global_learning_rate *= ratio;
    }
    inline static float ThresholdL1(float w) {
        if (w > +__global_lambdaL1) return - __global_lambdaL1;
        if (w < -__global_lambdaL1) return + __global_lambdaL1;
        return 0.0;
    }
    static size_t __global_minibatch_size;
    static float __global_learning_rate;
    static float __global_ema_rate;
    static float __global_sparse_rate;
    static float __global_lambdaL2, __global_lambdaL1;
    
    static bool __global_bTraining; // Tag the phase TRAIN or TEST
};

class DropoutUpdater : public GradientUpdater {
public:
    explicit DropoutUpdater(float _dropout_rate) : dropout_rate(_dropout_rate) {}
    inline void Mask(bool* dropout_mask, size_t len) {
        assert(dropout_mask);
        assert(dropout_rate > 0 && dropout_rate < 1);
        if (GradientUpdater::__global_bTraining == false) {
            fill(dropout_mask, dropout_mask + len, true);
            return;
        }
        for (size_t i = 0; i < len; i++) {
            dropout_mask[i] = SampleBinary(1.0 - dropout_rate);
        }
    }
    inline float rescale() {
        if (GradientUpdater::__global_bTraining == false) {
            return 1.0;
        }
        return 1.0 / (1.0 - dropout_rate);
    }
    float dropout_rate;
};

class SimpleUpdater : public GradientUpdater {
public:
    void learnable_params_cnt(size_t cnt) {
        __adagrad_params_cnt = cnt;
    }
    void update(size_t offset, size_t len, float* weight, float* grad) {
        avx_vecScalerAdd(weight, grad, weight,
                         -__global_learning_rate / __global_minibatch_size, len);
        memset(grad, 0, len * sizeof(float));
    }
    void update(size_t offset, vector<Matrix*>& weight, vector<Matrix*>& grad) {
        for (size_t i = 0; i < weight.size(); i++) {
            weight[i]->subtract(grad[i], __global_learning_rate / __global_minibatch_size);
            grad[i]->zeroInit();
        }
    }
private:
    size_t __adagrad_params_cnt;
};

class AdagradUpdater : public GradientUpdater {
public:
    void learnable_params_cnt(size_t cnt) {
        __adagrad_params_cnt = cnt;
        __adagrad_accum.resize(cnt);
        cache = NULL;
    }
    void clear() {
        for (size_t i = 0; i < __adagrad_params_cnt; i++) {
            __adagrad_accum[i]->zeroInit();
        }
    }
    void update(size_t offset, vector<Matrix*>& weight, vector<Matrix*>& grad) {
        assert(weight.size() == grad.size());
        assert(offset + weight.size() <= __adagrad_params_cnt);
        for (size_t i = 0; i < weight.size(); i++) {
            if (cache) {
                cache->reshape(grad[i]->x_len, grad[i]->y_len);
            }
            cache = grad[i]->scale(1.0 / __global_minibatch_size)->copy(cache);
            cache->pow(2.0);
            cache->add(1e-12);
            
            if (__adagrad_accum[offset + i] == NULL) {
                __adagrad_accum[offset + i] = new Matrix(cache->x_len, cache->y_len);
                __adagrad_accum[offset + i]->zeroInit();
            }
            __adagrad_accum[offset + i]->add(cache);
            cache = __adagrad_accum[offset + i]->copy(cache)->pow(0.5);
            grad[i]->dotProduct(cache->inverse());
            weight[i]->subtract(grad[i], __global_learning_rate);
            grad[i]->zeroInit();
        }
    }
private:
    Matrix* cache;
    vector<Matrix*> __adagrad_accum;
    size_t __adagrad_params_cnt;
};

class AdagradUpdater_Num : public GradientUpdater {
public:
    void learnable_params_cnt(size_t cnt) {
        __adagrad_params_cnt = cnt;
        __adagrad_accum.resize(cnt);
        fill(__adagrad_accum.begin(), __adagrad_accum.end(), 0.0f);
    }
    void clear() {
        fill(__adagrad_accum.begin(), __adagrad_accum.end(), 0);
    }
    template<typename T>
    void update(size_t offset, size_t len, T* weight, T* grad) {
        assert(offset + len <= __adagrad_params_cnt);
        avx_vecScale(grad, grad, len, 1.0 / __global_minibatch_size);
        for (size_t i = 0; i < len; i++) {
            const float g = grad[i];
            if (g != 0) {
                __adagrad_accum[offset + i] += g * g;
                weight[i] -= __global_learning_rate * g / sqrt(__adagrad_accum[offset + i] + 1e-12);
            }
        }
        memset(grad, 0, len * sizeof(T));
    }
private:
    vector<float> __adagrad_accum;
    size_t __adagrad_params_cnt;
};

class RMSpropUpdater : public GradientUpdater {
public:
    void learnable_params_cnt(size_t cnt) {
        __rms_params_cnt = cnt;
        __rms_accum.resize(__rms_params_cnt);
        tmp = NULL;
    }
    void clear() {
        for (size_t i = 0; i < __rms_params_cnt; i++) {
            __rms_accum[i]->zeroInit();
        }
    }
    void update(size_t offset, vector<Matrix*>& weight, vector<Matrix*>& grad) {
        assert(weight.size() == grad.size());
        assert(offset + weight.size() <= __rms_params_cnt);
        for (size_t i = 0; i < weight.size(); i++) {
            if (tmp) {
                tmp->reshape(grad[i]->x_len, grad[i]->y_len);
            }
            tmp = grad[i]->scale(1.0 / __global_minibatch_size)->copy(tmp);
            tmp->pow(2.0);
            
            if (__rms_accum[offset + i] == NULL) {
                __rms_accum[offset + i] = new Matrix(tmp->x_len, tmp->y_len);
                __rms_accum[offset + i]->zeroInit();
            }
            __rms_accum[offset + i]->add(tmp, 1.0 - __global_ema_rate, __global_ema_rate);
            
            tmp = __rms_accum[offset + i]->copy(tmp);
            tmp->add(1e-12);
            tmp->pow(0.5);
            
            grad[i]->dotProduct(tmp->inverse());
            
            weight[i]->subtract(grad[i], __global_learning_rate);
            grad[i]->zeroInit();
        }
    }
private:
    Matrix *tmp;
    vector<Matrix*> __rms_accum;
    size_t __rms_params_cnt;
};

class RMSpropUpdater_Num : public GradientUpdater {
public:
    void learnable_params_cnt(size_t cnt) {
        __rms_params_cnt = cnt;
        __rms_accum.resize(__rms_params_cnt);
        fill(__rms_accum.begin(), __rms_accum.end(), 0.0f);
    }
    void clear() {
        for (size_t i = 0; i < __rms_params_cnt; i++) {
            __rms_accum[i] = 0;
        }
    }
    template<typename T>
    void update(size_t offset, size_t len, T* weight, T* grad) {
        assert(offset + len <= __rms_params_cnt);
        for (size_t i = 0; i < len; i++) {
            float g = grad[i] / __global_minibatch_size, tmp;
            if (g != 0) {
                // ema_rate closer to 1, EMA can smooth more elements (1-q^n)/(1-q)
                __rms_accum[offset + i] =
                    __rms_accum[offset + i] * __global_ema_rate +
                    (1.0 - __global_ema_rate) * g * g;
                tmp = 1.0 / (__rms_accum[offset + i] + 1e-12);
                g *= sqrt(tmp);
                
                weight[i] -= __global_learning_rate * g;
            }
            grad[i] = 0.0f;
        }
    }
private:
    vector<float> __rms_accum;
    size_t __rms_params_cnt;
};

class FTRLUpdater : public GradientUpdater { // Online Learning
public:
    ~FTRLUpdater() {
        delete [] ftrl_n;
        delete [] ftrl_sigma;
        delete [] ftrl_z;
    }
    void learnable_params_cnt(size_t cnt) {
        __ftrl_params_cnt = cnt;
        ftrl_n = new float[cnt];
        ftrl_sigma = new float[cnt];
        ftrl_z = new float[cnt];
        
        memset(ftrl_z, 0, sizeof(float) * cnt);
        memset(ftrl_n, 0, sizeof(float) * cnt);
        memset(ftrl_sigma, 0, sizeof(float) * cnt);
    }
    void update(size_t offset, size_t len, float*& weight, float*& grad) {
        assert(offset + len <= __ftrl_params_cnt);
        for (size_t fid = 0; fid < len; fid++) {
            if (grad[fid] == 0) {
                continue;
            }
            const float g2 = grad[fid] * grad[fid];
            ftrl_sigma[fid] = (sqrt(ftrl_n[fid] + g2) - sqrt(ftrl_n[fid])) / alpha;
            ftrl_z[fid] += grad[fid] - ftrl_sigma[fid] * weight[fid];
            ftrl_n[fid] += g2;
            if(fabs(ftrl_z[fid]) <= lambda1) {
                weight[fid] = 0.0f;
            } else {
                float tmpr = ftrl_z[fid];
                if(tmpr >= 0)
                    tmpr -= lambda1;
                else
                    tmpr += lambda1;
                weight[fid] = - tmpr / ((beta + sqrt(ftrl_n[fid])) / alpha + lambda2);
            }
        }
    }
private:
    const float alpha = 0.15f, lambda1 = 1.0f, beta = 1.0f, lambda2 = 1.0f;
    size_t __ftrl_params_cnt;
    float *ftrl_n, *ftrl_sigma, *ftrl_z;
};

#endif /* gradientUpdater_h */
