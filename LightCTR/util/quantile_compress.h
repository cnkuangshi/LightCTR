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

template <typename RealT>
class QuantileCompress {
    typedef char CompressT;
public:
    QuantileCompress(RealT _min, RealT _max) : min(_min), max(_max) {
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
                                 &QuantileCompress<RealT>::encoding,
                                 this,
                                 std::placeholders::_1
                                 )
                       );
    }
    void extract(const CompressT *input, const int len, RealT *output) {
        std::transform(input, input + len,
                       output,
                       std::bind(
                                 &QuantileCompress<RealT>::decoding,
                                 this,
                                 std::placeholders::_1
                                 )
                       );
    }
    
private:
    void init() {
        _delta = (max - min) / static_cast<RealT>(N_INTERVALS);
        for (int i = 0; i < N_INTERVALS; i++) {
            if (i == 0) {
                _real_value[i] = min;
            } else {
                _real_value[i] = _real_value[i - 1] + _delta;
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
            real -= min;
            ret = static_cast<CompressT>(real / _delta);
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
    
    static const size_t N_INTERVALS = 1 << (sizeof(CompressT) * 8);
    RealT min, max;
    RealT _delta;
    RealT _real_value[N_INTERVALS];
};

#endif /* quantile_compress_h */
