//
//  num_quantized.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/6/9.
//  Copyright © 2018年 SongKuangshi. All rights reserved.
//

#ifndef num_quantized_h
#define num_quantized_h

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <random>
#include "quantile_compress.h"
#include "../common/avx.h"

template <typename RealT>
RealT* lowbit_quantize(const RealT* vec, size_t len, int bitCount) {
    for (size_t i = 0; i < len; i++) {
        if (bitCount == 1) {
            // 1 bit, only care about sign of number
            vec[i] = vec[i] < 0 ? - 1.0 / 3 : 1.0 / 3;
        } else if (bitCount == 2) {
            // 2 bits, Boundaries: [0, 0.5]
            RealT tmp = fabs(vec[i]);
            if (tmp >= 0 && tmp <= 0.5) {
                vec[i] = 0.25;
            } else {
                vec[i] = 0.75;
            }
        } else {
            uint8_t *code = new uint8_t[100];
            QuantileCompress<float, uint8_t> compress(QuantileType::NORMAL_DISTRIBUT, -5, 5);
            compress.compress(vec, len, *code);
            compress.extract(*code, len, vec);
        }
    }
    return vec;
}

template <typename RealT, typename CompressT>
class Product_quantizer {
public:
    Product_quantizer(size_t _dimension, size_t _part_cnt, CompressT _cluster_cnt)
        : dimension(_dimension), part_cnt(_part_cnt), cluster_cnt(_cluster_cnt) {
            // by my experiment, 60 parts and 256 clusters show best performance
            assert(1 << 8 * sizeof(CompressT) >= static_cast<CompressT>(_cluster_cnt));
            assert(_dimension % _part_cnt == 0);
            centroids = new RealT[dimension * cluster_cnt];
            assert(centroids);
    }
    
    ~Product_quantizer() {
        delete[] centroids;
    }
    
    vector<vector<CompressT> > train(const RealT* data, size_t len) {
        assert(len >= cluster_cnt);
        
        size_t part_dim_interval = dimension / part_cnt;
        
        vector<vector<CompressT> > quantizated_codes;
        quantizated_codes.reserve(part_cnt);
        for (size_t i = 0; i < part_cnt; i++) {
            auto part_centroids = centroids + i * cluster_cnt * part_dim_interval;
            auto part_quancodes = std::vector<CompressT>(len);
            kmeans(data, len, i, part_centroids, part_dim_interval, part_quancodes);
            quantizated_codes.emplace_back(part_quancodes);
        }
        return quantizated_codes; // return local vector by RVO
    }
    
    RealT* get_centroids(size_t which_part, CompressT which_class) {
        size_t part_dim_interval = dimension / part_cnt;
        return centroids + which_part * cluster_cnt * part_dim_interval + \
                which_class * part_dim_interval;
    }
    
private:
    // get centroids of one part of dimension
    void kmeans(const RealT* data, size_t len,
                const size_t which_part, RealT* part_centroids,
                size_t sub_dim, std::vector<CompressT>& part_quancodes) {
        // random select one data row to be centroid
        std::vector<size_t> perm(len);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        
        for (size_t i = 0; i < cluster_cnt; i++) {
            auto init_centroid = data + perm[i] * dimension;
            auto x = init_centroid + which_part * sub_dim;
            memcpy(&part_centroids[i * sub_dim], x, sub_dim * sizeof(RealT));
        }
        // begin to train by kmeans
        float preIntera = 0x0fffffff;
        for (size_t i = 0; i < 100; i++) {
            auto intera = Estep(data, len, which_part, part_centroids,
                                part_quancodes.data(), sub_dim);
            if (fabs(preIntera - intera) < 1e-5) {
                break;
            }
            preIntera = intera;
            MStep(data, len, which_part, part_centroids, part_quancodes.data(), sub_dim);
        }
        printf("Finish part %zu intera = %f\n", which_part, preIntera);
    }
    
    float Estep(const RealT* data, size_t len,
               const size_t which_part, const RealT* part_centroids,
               CompressT* part_quancodes, size_t sub_dim) const {
        float interactive = 0;
        // loop all data rows
        for (size_t i = 0; i < len; i++) {
            auto row = data + i * dimension;
            auto x = row + which_part * sub_dim;
            // comparing with each centroids to find closest centroid
            RealT dis = avx_L2Distance(x, part_centroids, sub_dim);
            *part_quancodes = 0;
            for (size_t j = 1; j < cluster_cnt; j++) {
                RealT tmp = avx_L2Distance(x, part_centroids + j * sub_dim, sub_dim);
                if (tmp < dis) {
                    *part_quancodes = static_cast<CompressT>(j);
                    dis = tmp;
                }
            }
            part_quancodes += 1;
            interactive += dis;
        }
        return interactive;
    }
    
    void MStep(const RealT* data, size_t len,
               const size_t which_part, RealT* part_centroids,
               const CompressT* part_quancodes, size_t sub_dim) {
        
        std::vector<size_t> ele_cnt(cluster_cnt, 0);
        memset(part_centroids, 0, cluster_cnt * sub_dim * sizeof(RealT));

        for (size_t i = 0; i < len; i++) {
            auto row = data + i * dimension;
            auto x = row + which_part * sub_dim;
            
            auto which_class = static_cast<CompressT>(part_quancodes[i]);
            assert(which_class >= 0 && which_class < cluster_cnt);
            ele_cnt[which_class]++;
            
            auto c = part_centroids + which_class * sub_dim;
            avx_vecAdd(c, x, c, sub_dim);
        }
        
        for (auto k = 0; k < cluster_cnt; k++) {
            const size_t cnt = ele_cnt[k];
            auto c = part_centroids + k * sub_dim;
            if (cnt > 0) {
                avx_vecScale(c, c, sub_dim, 1.0 / cnt);
            }
        }
        
        std::uniform_real_distribution<> runiform(0, 1);
        for (auto k = 0; k < cluster_cnt; k++) {
            if (ele_cnt[k] == 0) {
                // select one cluster class to divided into two parts
                size_t m = 0;
                while (runiform(rng) * (len - cluster_cnt) >= ele_cnt[m] - 1) {
                    m = (m + 1) % cluster_cnt;
                }
                // copy selected centroids vector and modify small step
                memcpy(part_centroids + k * sub_dim,
                       part_centroids + m * sub_dim,
                       sizeof(RealT) * sub_dim);
                for (size_t j = 0; j < sub_dim; j++) {
                    int32_t sign = (j % 2) * 2 - 1;
                    centroids[k * sub_dim + j] += sign * 1e-5;
                    centroids[m * sub_dim + j] -= sign * 1e-5;
                }
                ele_cnt[k] = ele_cnt[m] / 2;
                ele_cnt[m] -= ele_cnt[k];
            }
        }
    }
    
    size_t dimension;
    size_t part_cnt;
    CompressT cluster_cnt;
    RealT* centroids;
    
    std::minstd_rand rng;
};


#endif /* num_quantized_h */
