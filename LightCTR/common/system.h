//
//  system.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/3.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef system_h
#define system_h

#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/shm.h>

#include "assert.h"
#include "lock.h"

#ifndef likely
#define likely(x)  __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x)  __builtin_expect(!!(x), 0)
#endif

inline int getEnv(const char *env_var, int defalt) {
    auto p = getenv(env_var);
    if (!p) {
        return defalt;
    }
    return atoi(p);
}

inline const char * getEnv(const char *env_var, const char *defalt) {
    auto p = getenv(env_var);
    if (!p) {
        return defalt;
    }
    return p;
}

template <class FUNC, class... ARGS>
auto ignore_signal_call(FUNC func, ARGS &&... args) ->
typename std::result_of<FUNC(ARGS...)>::type {
    for (;;) {
        auto err = func(args...);
        if (err < 0 && errno == EINTR) {
            puts("Ignored EINTR Signal, retry");
            continue;
        }
        return err;
    }
}

double SystemMemoryUsage() {
    FILE* fp = fopen("/proc/meminfo", "r");
    assert(fp);
    size_t bufsize = 256 * sizeof(char);
    char* buf = new (std::nothrow) char[bufsize];
    assert(buf);
    int totalMem = -1, freeMem = -1, bufMem = -1, cacheMem = -1;
    
    while (getline(&buf, &bufsize, fp) >= 0) {
        if (0 == strncmp(buf, "MemTotal", 8)) {
            if (1 != sscanf(buf, "%*s%d", &totalMem)) {
                std::cout << "failed to get MemTotal from string: [" << buf << "]";
            }
        } else if (0 == strncmp(buf, "MemFree", 7)) {
            if (1 != sscanf(buf, "%*s%d", &freeMem)) {
                std::cout << "failed to get MemFree from string: [" << buf << "]";
            }
        } else if (0 == strncmp(buf, "Buffers", 7)) {
            if (1 != sscanf(buf, "%*s%d", &bufMem)) {
                std::cout << "failed to get Buffers from string: [" << buf << "]";
            }
        } else if (0 == strncmp(buf, "Cached", 6)) {
            if (1 != sscanf(buf, "%*s%d", &cacheMem)) {
                std::cout << "failed to get Cached from string: [" << buf << "]";
            }
        }
        if (totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1) {
            break;
        }
    }
    assert(totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1);
    fclose(fp);
    delete[] buf;
    double usedMem = 1.0 - 1.0 * (freeMem + bufMem + cacheMem) / totalMem;
    return usedMem;
}

bool mmapLoad(const char* filename) {
    int _fd = open(filename, O_RDONLY, (int)0400);
    if (_fd == -1) {
        _fd = 0;
        return false;
    }
    off_t size = lseek(_fd, 0, SEEK_END);
    void* _nodes;
#ifdef MAP_POPULATE
    _nodes = mmap(
                  0, size, PROT_READ, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    _nodes = mmap(
                  0, size, PROT_READ, MAP_SHARED, _fd, 0);
#endif
    return true;
}

char* getShmAddr(int key, size_t size, int flag = 0666|IPC_CREAT) {
    assert(key != 0);
    
    int shmId = shmget(key, size, flag);
    if (shmId < 0) {
        // ipcs -m
        // sysctl -w kern.sysv.shmmax to adjust shm max memory size
        printf("%d %s\n", errno, strerror(errno));
    }
    assert(shmId >= 0);
    
    char* shmAddr = (char *)shmat(shmId, NULL, 0);
    assert(shmAddr != (char *)-1);
    
    return shmAddr;
}

#endif /* system_h */
