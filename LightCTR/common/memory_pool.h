//
//  memory_pool.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/26.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef memory_pool_h
#define memory_pool_h

#include <list>
#include <memory>
#include "lock.h"

// Memory Pool for managing vector allocation and deallocation
// Meanwhile, it can monitored memory leak and wild pointer
class MemoryPool {
public:
    static MemoryPool& Instance() { // singleton
        static MemoryPool pool;
        return pool;
    }
    
    ~MemoryPool() {
        leak_checkpoint();
        std::unique_lock<std::mutex> f_lock(freePtr_lock);
        for (auto& pair : freePtr_list) {
            free(pair.second);
        }
    }
    
    inline void leak_checkpoint() {
        assert(allocPtr_list.empty()); // memory leaks
    }
    
    inline void* allocate(size_t size) {
        assert((size & 7) == 0);
        {
            freePtr_lock.lock();
            for (auto it = freePtr_list.begin(); it != freePtr_list.end(); it++) {
                const size_t tmp_size = it->first;
                if (tmp_size >= size && tmp_size <= (size * 3) >> 1) {
                    void* tmp_ptr = it->second;
                    freePtr_list.erase(it);
                    freePtr_lock.unlock();
                    
                    std::unique_lock<std::mutex> a_lock(allocPtr_lock);
                    allocPtr_list.push_back(std::make_pair(tmp_size, tmp_ptr));
                    return tmp_ptr;
                }
            }
            freePtr_lock.unlock();
        }
        std::unique_lock<std::mutex> a_lock(allocPtr_lock);
        size = _alignedMemSize(size);
        void* tmp_ptr = malloc(size);
        assert(tmp_ptr); // out of memory
        allocPtr_list.push_back(std::make_pair(size, tmp_ptr));
        return tmp_ptr;
    }
    
    inline void deallocate(void* ptr) {
        assert(ptr);
        allocPtr_lock.lock();
        for (auto it = allocPtr_list.begin(); it != allocPtr_list.end(); it++) {
            if (it->second == ptr) {
                const size_t tmp_size = it->first;
                allocPtr_list.erase(it);
                allocPtr_lock.unlock();
                
                std::unique_lock<std::mutex> f_lock(freePtr_lock);
                freePtr_list.push_back(std::make_pair(tmp_size, ptr));
                return;
            }
        }
        allocPtr_lock.unlock();
        assert(false); // wild pointer
    }
    
private:
    static const int MemAlignment = 16;
    inline size_t _alignedMemSize(size_t size) const {
        return (size + MemAlignment - 1) & -MemAlignment;
    }

    
    std::list<std::pair<size_t, void*> > freePtr_list, allocPtr_list;
    std::mutex freePtr_lock, allocPtr_lock;
};


template <typename T>
class ArrayAllocator {
public:
    typedef T           value_type;
    typedef T*          pointer;
    typedef const T*    const_pointer;
    typedef T&          reference;
    typedef const T&    const_reference;
    typedef size_t      size_type;
    typedef ptrdiff_t   difference_type;
    
    template <typename U>
    struct rebind {
        typedef std::allocator<U> other;
    };
    
    pointer allocate(size_type n, const void* hint=0) {
        return (T*)MemoryPool::Instance().allocate((difference_type)n * sizeof(T));
    }
    
    void deallocate(pointer p, size_type n) {
        MemoryPool::Instance().deallocate(p);
    }
    
    void destroy(pointer p) {
        p->~T();
    }
    
    pointer address(reference x) {
        return (pointer)&x;
    }
    
    const_pointer address(const_reference x) {
        return (const_pointer)&x;
    }
    
    size_type max_size() const {
        return size_type(UINTMAX_MAX / sizeof(T));
    }
};

#endif /* memory_pool_h */
