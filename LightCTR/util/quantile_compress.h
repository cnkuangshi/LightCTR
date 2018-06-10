//
//  quantile_compress.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/5/4.
//  Copyright © 2018年 SongKuangshi. All rights reserved.
//

#ifndef quantile_compress_h
#define quantile_compress_h

#include <algorithm>
#include <functional>
#include "significance.h"

enum QuantileType {
    UNIFORM = 0,
    LOG,
    NORMAL_DISTRIBUT, // parameters usually obey the normal law
    CUSTOM_DISTRIBUT
};

template <typename RealT, typename CompressT>
class QuantileCompress {
public:
    QuantileCompress(QuantileType _quantileType, RealT _min, RealT _max,
                     RealT _mu = 0, RealT _sigma = 1) :
    quantileType(_quantileType), min(_min), max(_max), mu(_mu), sigma(_sigma) {
        assert(_min < _max);
        init();
    }
    // Disable the copy and assignment operator
    QuantileCompress(const QuantileCompress &) = delete;
    QuantileCompress(QuantileCompress &&) = delete;
    QuantileCompress &operator=(const QuantileCompress &) = delete;
    QuantileCompress &operator=(QuantileCompress &&) = delete;
    
    void compress(const RealT *input, const int len, CompressT *output) {
        std::transform(input, input + len,
                       output,
                       std::bind(
                                 &QuantileCompress<RealT, CompressT>::encoding,
                                 this,
                                 std::placeholders::_1
                                 )
                       );
    }
    void extract(const CompressT *input, const int len, RealT *output) {
        std::transform(input, input + len,
                       output,
                       std::bind(
                                 &QuantileCompress<RealT, CompressT>::decoding,
                                 this,
                                 std::placeholders::_1
                                 )
                       );
    }
    
private:
    RealT convert(RealT x) {
        if (quantileType == QuantileType::LOG) {
            x = log(x);
        } else if (quantileType == QuantileType::NORMAL_DISTRIBUT) {
            x = StandardCDF(x);
        } else if (quantileType == QuantileType::CUSTOM_DISTRIBUT) {
            x = CustomCDF(x, mu, sigma);
        }
        return x;
    }
    
    void init() {
        if (quantileType == QuantileType::LOG) {
            assert(-min == max);
            minCDF = convert(1e-4), maxCDF = convert(max); // fix min if quantile by log
        } else {
            minCDF = convert(min), maxCDF = convert(max);
        }
        assert(maxCDF > minCDF);
        
        _delta = (maxCDF - minCDF) / static_cast<RealT>(N_INTERVALS);
        if (quantileType == QuantileType::LOG) {
            _delta *= 2.0f; // divided by positive and negative parts
        }
        
        if (quantileType == QuantileType::UNIFORM) {
            _real_value[0] = min;
            for (int i = 1; i < N_INTERVALS; i++) {
                _real_value[i] = _real_value[i - 1] + _delta;
            }
        } else if (quantileType == QuantileType::LOG) {
            const size_t half_size = N_INTERVALS >> 1;
            for (int i = 0; i < half_size; i++) {
                _real_value[half_size + i] = exp(minCDF + i * _delta);
                _real_value[half_size - i - 1] = - _real_value[half_size + i];
            }
        } else if (quantileType == QuantileType::NORMAL_DISTRIBUT) {
            _real_value[0] = min;
            for (int i = 1; i < N_INTERVALS; i++) {
                _real_value[i] = ReverseCDF(minCDF + i * _delta, 0, 1);
            }
        } else if (quantileType == QuantileType::CUSTOM_DISTRIBUT) {
            _real_value[0] = min;
            for (int i = 1; i < N_INTERVALS; i++) {
                _real_value[i] = ReverseCDF(minCDF + i * _delta, mu, sigma);
            }
        }
    }
    
    CompressT encoding(RealT real) const {
        CompressT ret = CompressT();
        if (real <= min) {
            ret = static_cast<CompressT>(0);
        } else if (real >= max) {
            ret = static_cast<CompressT>(N_INTERVALS - 1);
        } else {
            if (quantileType == QuantileType::UNIFORM) {
                real -= min;
                ret = static_cast<CompressT>(real / _delta);
            } else if (quantileType == QuantileType::LOG ||
                       quantileType == QuantileType::NORMAL_DISTRIBUT ||
                       quantileType == QuantileType::CUSTOM_DISTRIBUT) {
                ret = static_cast<CompressT>(_binary_search(real));
            }
        }
        return ret;
    }
    
    RealT decoding(CompressT comp) const {
        int index = static_cast<int>(comp);
        if (index < 0) { // deal with big-endian number
            index = N_INTERVALS + index;
        }
        assert(index >= 0 && index < N_INTERVALS);
        return _real_value[static_cast<size_t>(index)];
    }
    
    int _binary_search(RealT value) const {
        int lower = 0, upper = N_INTERVALS - 1, mid;
        while (lower <= upper) {
            mid = (lower + upper) >> 1;
            if (_real_value[mid] > value) {
                upper = mid - 1;
            } else {
                lower = mid + 1;
            }
        }
        return upper;
    }
    
    QuantileType quantileType;
    
    static const size_t N_INTERVALS = 1 << (sizeof(CompressT) * 8);
    RealT min, max;
    RealT minCDF, maxCDF;
    RealT mu, sigma;
    RealT _delta;
    RealT _real_value[N_INTERVALS];
};

#endif /* quantile_compress_h */
