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
#include "assert.h"

#ifndef likely
#define likely(x)  __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x)  __builtin_expect(!!(x), 0)
#endif

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

#endif /* system_h */
