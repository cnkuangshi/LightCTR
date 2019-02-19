//
//  buffer_fusion.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/1/1.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef buffer_fusion_h
#define buffer_fusion_h

#include "buffer.h"
#include <string.h>
#include <memory>

template <typename T>
class BufferFusion {
public:
    BufferFusion(bool _autoRelease, bool _lazyMode):
        autoRelease(_autoRelease), lazyMode(_lazyMode) {
        
    }
    
    ~BufferFusion() {
        if (autoRelease && lazyMode && lazyModeMemory) {
            delete lazyModeMemory;
            lazyModeMemory = nullptr;
            bufs_ptr_arr.clear();
            bufs_size_arr.clear();
            return;
        }
        if (!autoRelease) {
            bufs_ptr_arr.clear();
            bufs_size_arr.clear();
            return;
        }
        for (size_t i = 0; i < bufs_ptr_arr.size(); i++) {
            if (bufs_ptr_arr[i]) {
                delete[] bufs_ptr_arr[i];
                bufs_ptr_arr[i] = NULL;
            }
        }
        bufs_ptr_arr.clear();
        bufs_size_arr.clear();
    }
    
    std::pair<T*, size_t> getMemory(size_t index) {
        assert(index < bufs_size_arr.size());
        return std::make_pair(bufs_ptr_arr[index], bufs_size_arr[index]);
    }
    
    void registMemChunk(T* ptr, size_t size) {
        assert(size > 0);
        if (ptr != nullptr) {
            bufs_ptr_arr.push_back(ptr);
            bufs_size_arr.push_back(size);
            total_size += size;
        } else {
            assert(lazyMode);
            // lazy mode
            bufs_size_arr.push_back(size);
            total_size += size;
        }
    }
    
    void lazyAllocate(float* allocatedMem = nullptr) {
        if (allocatedMem) {
            lazyModeMemory = allocatedMem;
        } else {
            lazyModeMemory = new T[total_size];
        }
        size_t inc_mem = 0;
        for (size_t i = 0; i < bufs_size_arr.size(); i++) {
            bufs_ptr_arr.push_back(lazyModeMemory + inc_mem);
            inc_mem += bufs_size_arr[i];
        }
        assert(inc_mem == total_size);
    }
    
    size_t size() const {
        return total_size;
    }
    
    void memset_c(T __c) {
        if (likely(__c == 0)) {
            for (size_t i = 0; i < bufs_ptr_arr.size(); i++) {
                memset(bufs_ptr_arr[i], 0, bufs_size_arr[i] * sizeof(T));
            }
        } else {
            for (size_t i = 0; i < bufs_ptr_arr.size(); i++) {
                for (size_t j = 0; j < bufs_size_arr[i]; j++) {
                    *(bufs_ptr_arr[i] + j) = __c;
                }
            }
        }
    }
    
    void memcpy_out(Buffer** __dst, size_t __offset, size_t __n) const {
        assert(__offset + __n <= total_size);
        *__dst = new Buffer(__n);
        
        size_t which_one = 0;
        while (__offset >= bufs_size_arr[which_one]) {
            __offset -= bufs_size_arr[which_one];
            which_one++;
        }
        const T* __src = bufs_ptr_arr[which_one] + __offset;
        if (__n <= bufs_size_arr[which_one] - __offset) {
            (*__dst)->append(__src, __n * sizeof(T));
            return;
        }
        size_t offset = bufs_size_arr[which_one] - __offset;
        (*__dst)->append(__src, offset * sizeof(T));
        __n -= offset;
        
        size_t tmp = bufs_size_arr[++which_one];
        while (__n >= tmp) {
            (*__dst)->append(bufs_ptr_arr[which_one], tmp * sizeof(T));
            __n -= tmp;
            tmp = bufs_size_arr[++which_one];
        }
        if (__n > 0) {
            (*__dst)->append(bufs_ptr_arr[which_one], __n * sizeof(T));
        }
    }
    
    void memcpy_in(size_t __offset, const T* __src, size_t __n) {
        assert(__offset + __n <= total_size);
        size_t which_one = 0;
        while (__offset >= bufs_size_arr[which_one]) {
            __offset -= bufs_size_arr[which_one];
            which_one++;
        }
        T* __dst = bufs_ptr_arr[which_one] + __offset;
        if (__n <= bufs_size_arr[which_one] - __offset) {
            memcpy(__dst, __src, __n * sizeof(T));
            return;
        }
        size_t offset = bufs_size_arr[which_one] - __offset;
        memcpy(__dst, __src, offset * sizeof(T));
        __n -= offset;
        
        size_t tmp = bufs_size_arr[++which_one];
        while (__n >= tmp) {
            memcpy(bufs_ptr_arr[which_one], __src + offset, tmp * sizeof(T));
            __n -= tmp;
            offset += tmp;
            tmp = bufs_size_arr[++which_one];
        }
        if (__n > 0) {
            memcpy(bufs_ptr_arr[which_one], __src + offset, __n * sizeof(T));
        }
    }
    
    typedef std::function<void(T*, T*)> transform_callback_t;
    
    void transform(size_t __offset, size_t __n, transform_callback_t cb) const {
        assert(__offset + __n <= total_size);
        size_t which_one = 0;
        while (__offset >= bufs_size_arr[which_one]) {
            __offset -= bufs_size_arr[which_one];
            which_one++;
        }
        T* __dst = bufs_ptr_arr[which_one] + __offset;
        if (__n <= bufs_size_arr[which_one] - __offset) {
            cb(__dst, __dst + __n);
            return;
        }
        size_t offset = bufs_size_arr[which_one] - __offset;
        cb(__dst, __dst + offset);
        __n -= offset;
        
        size_t tmp = bufs_size_arr[++which_one];
        while (__n >= tmp) {
            cb(bufs_ptr_arr[which_one], bufs_ptr_arr[which_one] + tmp);
            __n -= tmp;
            tmp = bufs_size_arr[++which_one];
        }
        if (__n > 0) {
            cb(bufs_ptr_arr[which_one], bufs_ptr_arr[which_one] + __n);
        }
    }
    
    void flatten(Buffer** __dst) const {
        assert(total_size > 0);
        memcpy_out(__dst, 0, total_size);
    }
    
private:
    bool autoRelease{false};
    bool lazyMode{false};
    T* lazyModeMemory = nullptr;
    
    std::vector<T*> bufs_ptr_arr;
    std::vector<size_t> bufs_size_arr;
    size_t total_size = 0;
};

#endif /* buffer_fusion_h */
