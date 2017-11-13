//
//  gradientUpdater.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/23.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef gradientUpdater_h
#define gradientUpdater_h

#include "matrix.h"

class GradientUpdater {
public:
    // update weight with L2 Regularization
    inline static void update(double* weight, double grad) {
        *weight += grad + __global_lambdaL2 * *weight;
    }
    inline static void updateL1(double* weight, double grad) {
        *weight += grad + ThresholdL1(*weight);
    }
    inline static void update(vector<double>::iterator weight, double grad) {
        *weight += grad + __global_lambdaL2 * *weight;
    }
    inline static void decay(double ratio) {
        __global_learning_rate *= ratio;
    }
    inline static double ThresholdL1(double w) {
        if (w > +__global_lambdaL1) return - __global_lambdaL1;
        if (w < -__global_lambdaL1) return + __global_lambdaL1;
        return 0.0;
    }
    static size_t __global_minibatch_size;
    static double __global_learning_rate;
    static double __global_sparse_rate;
    static double __global_lambdaL2, __global_lambdaL1;
    
    static bool __global_bTraining; // Tag the phase TRAIN or TEST
};

class DropoutUpdater : public GradientUpdater {
public:
    explicit DropoutUpdater(double _dropout_rate) : dropout_rate(_dropout_rate) {}
    inline void Mask(bool* dropout_mask, size_t len) {
        assert(dropout_mask);
        assert(dropout_rate > 0 && dropout_rate < 1);
        for (size_t i = 0; i < len; i++) {
            dropout_mask[i] = SampleBinary(1.0 - dropout_rate);
        }
    }
    inline double rescale() {
        return 1.0 / (1.0 - dropout_rate);
    }
    double dropout_rate;
};

class SimpleUpdater : public GradientUpdater {
public:
    void learnable_params_cnt(size_t cnt) {
        __adagrad_params_cnt = cnt;
    }
    void update(size_t offset, size_t len, double*& weight, double*& grad) {
        for (size_t i = 0; i < len; i++) {
            weight[i] -= __global_learning_rate * grad[i] / __global_minibatch_size;
            grad[i] = 0.0f;
        }
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
        for (size_t i = 0; i < __adagrad_params_cnt; i++) {
            __adagrad_accum[i] = 0;
        }
    }
    template<typename T>
    void update(size_t offset, size_t len, T& weight, T& grad) {
        assert(offset + len <= __adagrad_params_cnt);
        for (size_t i = 0; i < len; i++) {
            double g = grad[i] / __global_minibatch_size;
            if (g != 0) {
                __adagrad_accum[offset + i] += g * g;
                weight[i] -= __global_learning_rate * g / sqrt(__adagrad_accum[offset + i] + 1e-12);
            }
            grad[i] = 0.0f;
        }
    }
private:
    vector<double> __adagrad_accum;
    size_t __adagrad_params_cnt;
};

class FTRLUpdater : public GradientUpdater { // Online Learning
public:
    ~FTRLUpdater() {
        delete [] ftrl_n;
        delete [] ftrl_sigma;
        delete [] ftrl_z;
    }
    void learnable_params_cnt(size_t cnt) {
        __adagrad_params_cnt = cnt;
        ftrl_n = new double[cnt];
        ftrl_sigma = new double[cnt];
        ftrl_z = new double[cnt];
        
        memset(ftrl_z, 0, sizeof(double) * cnt);
        memset(ftrl_n, 0, sizeof(double) * cnt);
        memset(ftrl_sigma, 0, sizeof(double) * cnt);
    }
    void update(size_t offset, size_t len, double*& weight, double*& grad) {
        assert(offset + len <= __adagrad_params_cnt);
        double alpha = 2.0f, lambda1 = 5.0f, beta = 1.0f, lambda2 = 0.0f;
        for (size_t fid = 0; fid < len; fid++) {
            if (grad[fid] == 0) {
                continue;
            }
            double g2 = grad[fid] * grad[fid];
            ftrl_sigma[fid] = (sqrt(ftrl_n[fid] + g2) - sqrt(ftrl_n[fid])) / alpha;
            ftrl_z[fid] += grad[fid] - ftrl_sigma[fid] * weight[fid];
            ftrl_n[fid] += g2;
            if(fabs(ftrl_z[fid]) <= lambda1) {
                weight[fid] = 0.0f;
            } else {
                double tmpr = 0.0f;
                if(ftrl_z[fid] >= 0)
                    tmpr = ftrl_z[fid] - lambda1;
                else
                    tmpr = ftrl_z[fid] + lambda1;
                weight[fid] = - tmpr / ((beta + sqrt(ftrl_n[fid])) / alpha + lambda2);
            }
        }
    }
private:
    size_t __adagrad_params_cnt;
    double *ftrl_n, *ftrl_sigma, *ftrl_z;
};

#endif /* gradientUpdater_h */
