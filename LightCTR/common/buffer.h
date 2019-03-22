//
//  buffer.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/12.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef buffer_h
#define buffer_h

#include "float16.h"
#include <cstring>

class Buffer {
public:
    explicit Buffer(size_t capacity = 64) {
        assert(capacity > 0);
        _capacity = ((capacity + __align - 1) & (~(__align - 1)));
        _buffer = new char[_capacity];
        _cursor = _end = _buffer;
    }
    
    Buffer(const void* buf, size_t len) {
        assert(buf && len >= 0); // allow empty Buffer for Heartbeat
        _capacity = len;
        _buffer = new char[_capacity];
        _cursor = _end = _buffer;
        
        std::memcpy(_buffer, buf, len);
        _end += len;
    }
    
    Buffer(const Buffer &) = delete;
    Buffer(Buffer&& other) {
        if (this != &other) {
            free();
            _buffer = other._buffer;
            other._buffer = nullptr;
            _cursor = other._cursor;
            _end = other._end;
            _capacity = other._capacity;
        }
    }
    
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer&& other) {
        if (this != &other) {
            free();
            _buffer = other._buffer;
            other._buffer = nullptr;
            _cursor = other._cursor;
            _end = other._end;
            _capacity = other._capacity;
        }
        return *this;
    }
    
    ~Buffer() {
        free();
    }
    inline void free() {
        if (_buffer) {
            delete[] _buffer;
        }
        _buffer = _cursor = _end = nullptr;
        _capacity = 0;
    }
    
    inline void reset() {
        _cursor = _end = _buffer;
    }
    inline const char* buffer() const {
        return _buffer;
    }
    inline const char* cursor() const {
        return _cursor;
    }
    inline const char* end() const {
        return _end;
    }
    
    inline size_t capacity() const {
        return _capacity;
    }
    inline size_t size() const {
        return _end - _buffer;
    }
    inline bool empty() const {
        return size() == 0;
    }
    
    template <typename T>
    inline void append(const T* x, size_t len) {
        if (_capacity == 0) {
            _capacity = len;
            reserve(_capacity);
        } else {
            const size_t cur_size = size();
            size_t new_cap = _capacity;
            while (cur_size + len > new_cap) {
                new_cap = 2 * new_cap;
                new_cap = ((new_cap + __align - 1) & (~(__align - 1)));
            }
            reserve(new_cap);
        }
        assert(size() + len <= _capacity); // check address sanitizer
        std::memcpy(_end, x, len);
        _end += len;
    }
    
    template <typename T>
    inline void appendVarUint(T x) {
        const size_t type_size = sizeof(T) * 8;
        assert(type_size == 32 || type_size == 64);
        assert(x >= 0);
        
        char* beginPtr = new char[5];
        char* ptr = beginPtr;
        static const uint32_t B = 128;
        while (x >= B) {
            *(ptr++) = (x & (B-1)) | B;
            x >>= 7;
        }
        *(ptr++) = static_cast<char>(x);
        append(beginPtr, ptr - beginPtr);
        delete[] beginPtr;
    }
    
    template <typename T>
    inline void read(T* x, size_t len = 0) {
        if (len == 0) {
            len = size(); // read all
        }
        std::memcpy(x, _cursor, len);
        _cursor += len;
        assert(_cursor <= _end);
    }
    
    template <typename T>
    inline void readHalfFloat(T* x) {
        float16_t t;
        std::memcpy(&t, _cursor, sizeof(float16_t));
        _cursor += sizeof(float16_t);
        assert(_cursor <= _end);
        if (sizeof(T) == 4 || sizeof(T) == 8) {
            *x = static_cast<T>(Float16(t).float32_value());
        }
    }
    
    template <typename T>
    inline void readVarUint(T* x) {
        const size_t type_size = sizeof(T) * 8;
        assert(type_size == 32 || type_size == 64);
        T res = 0;
        
        bool check_flg = false;
        char *local_cursor = _cursor;
        for (size_t shift = 0; shift < type_size && local_cursor <= _end; shift += 7) {
            size_t byte = *(reinterpret_cast<const char*>(local_cursor)); // read one byte
            local_cursor++;
            if (byte & 128) {
                res |= (byte & 127) << shift;
            } else {
                res |= byte << shift;
                check_flg = true;
                break;
            }
        }
        assert(check_flg && res >= 0);
        *x = res;
        _cursor = local_cursor;
    }
    
    inline size_t readed_size() const {
        return _cursor - _buffer;
    }
    
    inline void reset_cursor() {
        _cursor = _buffer;
    }
    
    inline bool readEOF() const {
        assert(_cursor <= _end);
        return _cursor == _end;
    }
    inline void cursor_preceed(size_t size) {
        assert(_cursor + size <= _end);
        _cursor += size;
    }
    
    template <typename T>
    Buffer& operator >> (T& x) { // just for read
        read((char *)&x, sizeof(T));
        return *this;
    }
    template <typename T>
    Buffer& operator << (const T& x) {
        append(&x, sizeof(T));
        return *this;
    }
    
protected:
    inline void reserve(size_t newcap) {
        if (newcap > _capacity) {
            char* newbuf = new char[newcap];
            assert(newbuf);
            if (size() > 0) {
                std::memcpy(newbuf, _buffer, size());
            }
            
            const size_t cursor_offset = readed_size();
            const size_t old_size = size();
            free();
            
            _buffer = newbuf;
            _cursor = newbuf + cursor_offset;
            _end = newbuf + old_size;
            _capacity = newcap;
        }
    }
    
private:
    char *_buffer = nullptr;
    char *_cursor = nullptr;
    char *_end = nullptr;
    size_t _capacity;
    
    const size_t __align = 2;
};

#endif /* buffer_h */
