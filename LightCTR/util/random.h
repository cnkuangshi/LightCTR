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

inline double UniformNumRand() { // [0, 1)
    return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0);
}

inline double UniformNumRand2() { // (0, 1)
    return (static_cast<double>(rand()) + 1.0) / (static_cast<double>(RAND_MAX) + 2.0);
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

#endif /* random_h */
