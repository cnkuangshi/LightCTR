//
//  buffer.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/12.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef buffer_h
#define buffer_h

class Buffer {
public:
    Buffer() {
        assert(_capacity > 0);
        _buffer = new char[_capacity];
        _cursor = _end = _buffer;
    }
    Buffer(const void* buf, size_t len) {
        assert(_capacity > 0 && buf && len >= 0);
        _capacity = len;
        _buffer = new char[_capacity];
        _cursor = _end = _buffer;
        
        memcpy(_buffer, buf, len);
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
    Buffer &operator=(Buffer &&other) {
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
    inline char *buffer() const {
        return _buffer;
    }
    inline char *cursor() const {
        return _cursor;
    }
    inline char *end() const {
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
    inline void append(T *x, size_t len) {
        if (_capacity == 0) {
            _capacity = len;
            reserve(_capacity);
        } else {
            const size_t cur_size = size();
            size_t new_cap = _capacity;
            while (cur_size + len > new_cap) {
                new_cap = 2 * new_cap;
            }
            reserve(new_cap);
        }
        assert(size() + len <= _capacity); // check address sanitizer
        memcpy(_end, x, len);
        _end += len;
    }
    
    template <typename T>
    inline void read(T *x, size_t len = 0) {
        if (len == 0) {
            len = size(); // read all
        }
        memcpy(x, _cursor, len);
        _cursor += len;
        assert(_cursor <= _end);
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
    Buffer& operator >> (T &x) { // just for read
        read((char *)&x, sizeof(T));
        return *this;
    }
    template <typename T>
    Buffer& operator << (T &x) {
        append(&x, sizeof(T));
        return *this;
    }
    
protected:
    inline void reserve(size_t newcap) {
        if (newcap > _capacity) {
            char* newbuf = new char[newcap];
            assert(newbuf);
            if (size() > 0) {
                memcpy(newbuf, _buffer, size());
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
    size_t _capacity = 64;
};

#endif /* buffer_h */
