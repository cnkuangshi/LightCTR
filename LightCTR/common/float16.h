//
//  float16.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/12/3.
//  Copyright © 2018 SongKuangshi. All rights reserved.
//

#ifndef float16_h
#define float16_h

#define float16_t unsigned short

class Float16 {
public:
    Float16() {
        assert(sizeof(float) * 8 == 32);
        assert(sizeof(float16_t) * 8 == 16);
    }
    
    explicit Float16(void* src) { // double or float
        _float32_value = *(float*)src;
        convert(_float32_value);
    }
    
    explicit Float16(float16_t src) {
        _float16_value = src;
        _float32_value = toFloat32(src);
    }
    
    inline float16_t float16_value() {
        return _float16_value;
    }
    inline float float32_value() {
        return _float32_value;
    }
    
    void convert2Float16(const float* input, float16_t* output, int len) {
        std::transform(input, input + len,
                       output,
                       std::bind(
                                 &Float16::convert,
                                 this,
                                 std::placeholders::_1
                                 )
                       );
    }
    
    void recover2Float32(const float16_t* input, float* output, int len) {
        std::transform(input, input + len,
                       output,
                       std::bind(
                                 &Float16::toFloat32,
                                 this,
                                 std::placeholders::_1
                                 )
                       );
    }
    
private:
    inline float toFloat32(float16_t h) {
        int sign = ((h >> 15) & 1); // 1
        int exp = ((h >> 10) & 0x1f); // 5
        int mantissa = (h & 0x3ff); // 10
        unsigned f = 0;
        
        if (exp > 0 && exp < 31) {
            // normal
            exp += 112; // 127 - 15
            f = (sign << 31) | (exp << 23) | (mantissa << 13);
        } else if (exp == 0) {
            if (mantissa) {
                // subnormal
                exp += 113; // 127 - 15 + 1
                while ((mantissa & (1 << 10)) == 0) {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= 0x3ff;
                f = (sign << 31) | (exp << 23) | (mantissa << 13);
            } else {
                f = (sign << 31); // ±0.0
            }
        } else if (exp == 31) {
            if (mantissa) {
                f = 0x7fffffff;  // NAN
            } else {
                f = (0xff << 23) | (sign << 31);  // INF
            }
        }
#ifdef DEBUG
        printf("%f\n", *reinterpret_cast<float*>(&f));
        print_bin16((float16_t)h);
        print_bin(*reinterpret_cast<float*>(&f));
        puts("");
#endif
        return *reinterpret_cast<float*>(&f);
    }
    
    inline float16_t convert(float src) {
        // convert Float32 into Binary float16 (unsigned short) based IEEE754 standard
        unsigned const& s = *reinterpret_cast<unsigned const*>(&src);

        uint16_t sign = uint16_t((s >> 16) & 0x8000); // 1
        int16_t exp = uint16_t(((s >> 23) & 0xff) - 127); // 8
        int mantissa = s & 0x7fffff; // 23
    
        if ((s & 0x7fffffff) == 0) { // ±0.0
            return 0;
        }
        // special number
        if (exp > 15) { // bias changes from 127 to 15
            if (exp == 128 && mantissa) {
                // still NAN
                return 0x7fff;
            } else {
                // exp > 15 causes upper overflow, INF
                return sign | 0x7c00;
            }
        }
    
        uint16_t u = 0;
        int sticky_bit = 0;
    
        if (exp >= -14) {
            // normal fp32 to normal fp16
            exp = uint16_t(exp + uint16_t(15));
            u = uint16_t(((exp & 0x1f) << 10));
            u = uint16_t(u | (mantissa >> 13));
        } else {
            // normal float to subnormal (exp=0)
            int rshift = - (exp + 14);
            if (rshift < 32) {
                mantissa |= (1 << 23);
                sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);
                
                mantissa = (mantissa >> rshift);
                u = (uint16_t(mantissa >> 13) & 0x3ff);
            } else {
                // drop precision
                mantissa = 0;
                u = 0;
            }
        }
    
        // round to nearest even
        int round_bit = ((mantissa >> 12) & 1);
        sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);
    
        if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
            u = uint16_t(u + 1);
        }
    
        u |= sign;
#ifdef DEBUG
        printf("%f\n", src);
        print_bin(src);
        print_bin16((float16_t)u);
        puts("");
#endif
        return *reinterpret_cast<float16_t*>(&u);
    }
    
    void print_bin(float num) {
        printf("32: ");
        unsigned const& s = *reinterpret_cast<unsigned const*>(&num);
        for(int i = 1; i <= sizeof(num) * 8; i++) {
            printf("%d", (s >> (sizeof(num) * 8 - i)) & 1);
            if (i == 1 || i == 9 || i == 32) {
                printf("\t");
            }
        }
        puts("");
    }
    void print_bin16(float16_t num) {
        printf("16: ");
        unsigned const& s = *reinterpret_cast<unsigned const*>(&num);
        for(int i = 1; i <= sizeof(num) * 8; i++) {
            printf("%d", (s >> (sizeof(num) * 8 - i)) & 1);
            if (i == 1 || i == 6 || i == 16) {
                printf("\t");
            }
        }
        puts("");
    }
    
    float16_t _float16_value;
    float _float32_value;
};

#endif /* float16_h */
