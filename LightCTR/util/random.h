//
//  random.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/24.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef random_h
#define random_h

#include <cmath>
#include <cstdio>
#include <vector>
#include "significance.h"

inline void Seed(uint32_t seed) {
    srand(seed);
}

inline double UniformNumRand() { // [0, 1)
    return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0);
}

inline double UniformNumRand2() { // (0, 1)
    return (static_cast<double>(rand()) + 1.0) / (static_cast<double>(RAND_MAX) + 2.0);
}

inline size_t Random_index(size_t n) {
    return rand() % n;
}

template<typename T>
inline void Shuffle(T *vec, size_t sz) {
    if (sz == 0)
        return;
    for (uint32_t i = (uint32_t)sz - 1; i > 0; i--) {
        std::swap(vec[i], vec[(uint32_t)(UniformNumRand() * (i + 1))]);
    }
}

inline double GaussRand() { // ~N(0, 1)
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if(phase == 0){
        do {
            V1 = 2.0 * UniformNumRand2() - 1.0;
            V2 = 2.0 * UniformNumRand2() - 1.0;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1.0 || S == 0.0);
        X = V1 * sqrt(-2.0 * log(S) / S);
    } else {
        X = V2 * sqrt(-2.0 * log(S) / S);
    }
    phase = 1 - phase;
    return X;
}

inline double GaussRand(double mu, double sigma) {
    return GaussRand() * sigma + mu;
}

inline std::pair<double, double> GaussRand2D() {
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if(phase == 0){
        do {
            V1 = 2.0 * UniformNumRand2() - 1.0;
            V2 = 2.0 * UniformNumRand2() - 1.0;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1.0 || S == 0.0);
        X = V1 * sqrt(-2.0 * log(S) / S);
    } else {
        X = V2 * sqrt(-2.0 * log(S) / S);
    }
    phase = 1 - phase;
    return std::make_pair(V1 * X, V2 * X);
}

inline bool SampleBinary(double p) {
    return UniformNumRand() < p;
}

inline size_t subSampleSize(double sampleAlpha = 0.05, double sampleErrorBound = 0.05) {
    // indicate confidence level and error bound to determine a suitable sample size
    double z = ReverseAlpha(sampleAlpha / 2);
    size_t sampleSize = (size_t)((z * z / 4.0f) / (sampleErrorBound * sampleErrorBound));
    // double minProb = 9.0f / (9.0f + sampleSize);
    // double maxProb = 1.0f * sampleSize / (9.0 + sampleSize);
    // sigma = sqrt(prob * (1 - prob) / sampleSize)
    // max(delta / sigma) determine the significance of distribution difference
    return sampleSize;
}

inline void shuffleSelectK(std::vector<size_t>* rankResult, size_t n, size_t k) {
    // when n equal to k mean shuffle, otherwise sample should adjust k
    assert(n / 2 >= k);
    if (rankResult->size() != k) {
        rankResult->clear();
        rankResult->resize(k);
    }
    std::vector<size_t> array;
    array.resize(n);
    for (size_t i = 0; i < n; i++) {
        array[i] = i;
    }
    for (size_t i = 0; i < k; i++) {
        size_t index = UniformNumRand() * (n - i);
        rankResult->at(i) = array[index];
        array[index] = array[n - 1 - i];
    }
}

#endif /* random_h */
