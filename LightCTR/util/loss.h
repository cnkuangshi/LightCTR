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
    virtual T loss(const vector<T>* pred, const vector<L>* label) const = 0;
    virtual void gradient(const vector<T>* pred, const vector<L>* label, vector<T>* gradient) = 0;
};

template <typename T, typename L>
class Square : public Loss<T, L> { // Mean Squared Error
public:
    T loss(const vector<T>* pred, const vector<L>* label) const {
        T sum = 0.0f, tmp;
        for (size_t i = 0; i < pred->size(); i++) {
            tmp = pred->at(i) - label->at(i);
            sum += tmp / 2 * tmp;
        }
        return sum;
    }
    void gradient(const vector<T>* pred, const vector<L>* label, vector<T>* gradient) {
        assert(pred->size() == label->size());
        for (size_t i = 0; i < pred->size(); i++) {
            gradient->at(i) = pred->at(i) - label->at(i);
        }
    }
};

template <typename T, typename L>
class Logistic : public Loss<T, L> {
public:
    T loss(const vector<T>* pred, const vector<L>* label) const {
        assert(pred->size() == label->size());
        T sum = 0.0f, p, l;
        for (size_t i = 0; i < pred->size(); i++) {
            p = pred->at(i);
            l = label->at(i);
            sum += (l - (p >= 0)) * p - log(1.0f + exp(p - 2.0f * (p >= 0) * p));
//            sum += label->at(i) * log(pred->at(i)) + (1.0f - label->at(i)) * log(1.0f - pred->at(i));
        }
        assert(!isnan(sum));
        return sum;
    }
    void gradient(const vector<T>* pred, const vector<L>* label, vector<T>* gradient) {
        assert(pred->size() == label->size());
        assert(gradient->size() == label->size());
        // Notice output activator must be sigmoid
        for (size_t i = 0; i < pred->size(); i++) {
            gradient->at(i) = pred->at(i) - label->at(i);
        }
    }
};

template <typename T, typename L>
class Logistic_Softmax : public Loss<T, L> {
public:
    T loss(const vector<T>* pred, const vector<L>* label) const {
        assert(pred->size() == label->size());
        T sum = 0.0f;
        for (size_t i = 0; i < pred->size(); i++) {
            if (label->at(i) == 1) {
                sum += log(pred->at(i));
            }
        }
        assert(!isnan(sum));
        return sum;
    }
    void gradient(const vector<T>* pred, const vector<L>* label, vector<T>* gradient) {
        assert(pred->size() == label->size());
        assert(gradient->size() == label->size());
        
        for (size_t i = 0; i < pred->size(); i++) {
            if (label->at(i) == 1) {
                gradient->at(i) = 1.0f - pred->at(i);
            } else {
                gradient->at(i) = - pred->at(i);
            }
        }
    }
};

#endif /* loss_h */
