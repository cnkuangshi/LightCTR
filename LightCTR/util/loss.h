//
//  loss.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/20.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef loss_h
#define loss_h

#include <cmath>
#include <vector>
#include "assert.h"
using namespace std;

template <typename T, typename L>
class Loss {
public:
    virtual T loss(const T* pred, const L* label, size_t len) const = 0;
    virtual void gradient(const T* pred, const L* label, T* gradient, size_t len) = 0;
};

template <typename T, typename L>
class Square : public Loss<T, L> { // Mean Squared Error
public:
    T loss(const T* pred, const L* label, size_t len) const {
        T sum = 0.0f, tmp;
        for (size_t i = 0; i < len; i++) {
            tmp = pred[i] - label[i];
            sum += tmp / 2 * tmp;
        }
        return sum;
    }
    void gradient(const T* pred, const L* label, T* gradient, size_t len) {
        for (size_t i = 0; i < len; i++) {
            gradient[i] = pred[i] - label[i];
        }
    }
};

template <typename T, typename L>
class Logistic : public Loss<T, L> {
public:
    T loss(const T* pred, const L* label, size_t len) const {
        T sum = 0.0f, p, l;
        for (size_t i = 0; i < len; i++) {
            p = pred[i];
            l = label[i];
            sum += (l - (p >= 0)) * p - log(1.0f + exp(p - 2.0f * (p >= 0) * p));
//            sum += label->at(i) * log(pred->at(i)) + (1.0f - label->at(i)) * log(1.0f - pred->at(i));
        }
        assert(!isnan(sum));
        return sum;
    }
    void gradient(const T* pred, const L* label, T* gradient, size_t len) {
        // Notice output activator must be sigmoid
        for (size_t i = 0; i < len; i++) {
            gradient[i] = pred[i] - label[i];
        }
    }
};

template <typename T, typename L>
class Logistic_Softmax : public Loss<T, L> {
public:
    T loss(const T* pred, const L* label, size_t len) const {
        T sum = 0.0f;
        for (size_t i = 0; i < len; i++) {
            if (label[i] == 1) {
                sum += log(pred[i]);
            }
        }
        assert(!isnan(sum));
        return sum;
    }
    void gradient(const T* pred, const L* label, T* gradient, size_t len) {
        for (size_t i = 0; i < len; i++) {
            if (label[i] == 1) {
                gradient[i] = 1.0f - pred[i];
            } else {
                gradient[i] = - pred[i];
            }
        }
    }
};

#endif /* loss_h */
