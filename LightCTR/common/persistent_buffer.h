//
//  persistent_buffer.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/12/21.
//  Copyright © 2018 SongKuangshi. All rights reserved.
//

#ifndef persistent_buffer_h
#define persistent_buffer_h

#include "buffer.h"
#include "system.h"

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#ifdef __APPLE__
#include <sys/uio.h>
#else
#include <sys/io.h>
#endif

class PersistentBuffer  {
public:
    PersistentBuffer(const char* path, size_t size, bool alarm_when_exist) {
        int flag = O_CREAT | O_RDWR;
        if (alarm_when_exist) {
            flag |= O_EXCL;
        }
        int _fd = open(path, flag, 0666);
        if (_fd < 0) {
            printf("open file errno = %d %s\n", errno, strerror(errno));
        }
        assert(_fd >= 0);
        
        _capacity = lseek(_fd, 0, SEEK_END);
        if (_capacity < size) {
            assert(ftruncate(_fd, size) == 0);
            lseek(_fd, 0, SEEK_END);
            _capacity = size;
        }
        assert(size <= _capacity);
            
        assert(close(_fd) == 0);
        
        _buffer = nullptr;
        assert(mmapLoad(path, (void**)&_buffer, true));
        
        assert(_buffer);
        memset(_buffer, 0, _capacity);
        
        _cursor = _end = _buffer;
    }
    
    ~PersistentBuffer() {
        if (_buffer) {
            munmap(_buffer, _capacity);
        }
    }
    
    inline size_t size() const {
        return _end - _buffer;
    }
    
    template <typename T>
    inline void write(T *x, size_t len) {
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
    
private:
    char *_buffer = nullptr;
    char *_cursor = nullptr;
    char *_end = nullptr;
    size_t _capacity;
};

#endif /* persistent_buffer_h */
