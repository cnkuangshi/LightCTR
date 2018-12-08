//
//  shm_hashtable.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/12/7.
//  Copyright © 2018 SongKuangshi. All rights reserved.
//

#ifndef shm_hashtable_h
#define shm_hashtable_h

#include "../common/system.h"
#include "../common/lock.h"
#include "../common/hash.h"
#include <vector>
#include <bitset>
#include <string>

template <typename T>
class ShmHashTable {
public:
    struct ShmHashNode {
        size_t key; // preserving zero for identifing empty
        T value;
        ShmHashNode() : key(0), value(0.0) {}
    };
    
    static ShmHashTable& Instance(size_t hash_times) {
        static ShmHashTable _instance(hash_times);
        return _instance;
    }
    
    bool insert(const std::string& key, const T& value) {
        return insert(static_cast<size_t>(murMurHash(key)), value);
    }
    
    bool update(const std::string& key, const T& value) {
        return update(static_cast<size_t>(murMurHash(key)), value);
    }
    
    bool update(size_t key, const T& value) {
        return insert(key, value);
    }
    
    bool insert(size_t key, const T& value) {
        assert(g_pShmAddr);
        assert(key > 0);
        int res = insertOrUpdate(key, value, 0);
        return (res == 0 ? true : false);
    }
    
private:
    ShmHashTable() {
        
    }
    ~ShmHashTable() {
        if (g_pShmAddr) {
            shmdt(g_pShmAddr);
            g_pShmAddr = NULL;
        }
    }
    explicit ShmHashTable(size_t _hash_times) {
        hash_times = _hash_times;
        tashtable_reserve_size = hashspace * _hash_times * sizeof(ShmHashNode);
        
        initPrime(primes);
        
        g_pShmAddr = getShmAddr(0x5fef, tashtable_reserve_size);
    }
    ShmHashTable(const ShmHashTable&) = delete;
    ShmHashTable(ShmHashTable&&) = delete;
    ShmHashTable& operator=(const ShmHashTable&) = delete;
    ShmHashTable& operator=(ShmHashTable&&) = delete;
    
    int insertOrUpdate(size_t key, T value, size_t depth) {
        if (depth > 5)
            return -1;
        
        vector<ShmHashNode*> candidate_position;
        candidate_position.reserve(hash_times);
        
        for (int i = 0; i < hash_times; i++) {
            size_t inner_offset = key % primes[i];
            ShmHashNode* addr = (ShmHashNode*)g_pShmAddr + prime_offset[i] + inner_offset;
            if (addr->key == 0) {
                candidate_position.emplace_back(addr);
            } else if (addr->key == key) {
                // update
                if(!atomic_compare_and_swap(&addr->value, addr->value, value)) {
                    return insertOrUpdate(key, value, depth + 1);
                }
            }
        }
        
        // select one empty slot to insert
        if (likely(!candidate_position.empty())) {
            for (int i = 0; i < candidate_position.size(); i++) {
                ShmHashNode* addr = candidate_position[i];
                if (addr->key == 0) {
                    unique_lock<SpinLock> glock(lock);
                    if (addr->key == 0) {
                        addr->key = key;
                        addr->value = value;
                        
                        return 0;
                    }
                }
            }
        }
        // conflict happened
        return insertOrUpdate(key, value, depth + 1);
    }
    
    void initPrime(std::vector<size_t>& primes) {
        static const size_t MAX = (hashspace >> 1) + 1;
        bitset<MAX> flag(0);
        
        primes.emplace_back(2);
        
        size_t i, j;
        for (i = 3; i < MAX; i += 2) {
            if (!(flag.test(i / 2)))
                primes.emplace_back(i);
            for (j = 1; j < primes.size() && i * primes[j] < MAX; j++) {
                flag.set(i * primes[j] / 2);
                if (i % primes[j] == 0)
                    break;
            }
        }
        std::reverse(primes.begin(), primes.end());
        primes.resize(hash_times);
        assert(primes.size() == hash_times);
        
        prime_offset.emplace_back(0);
        for (i = 0; i < hash_times; i++) {
            prime_offset.emplace_back(prime_offset.back() + primes[i]);
        }
    }
    
    void* g_pShmAddr = NULL;
    size_t hash_times;
    size_t tashtable_reserve_size = 0;
    
    static const size_t hashspace = 1 << 20;
    std::vector<size_t> primes;
    std::vector<size_t> prime_offset;
    
    SpinLock lock;
};

#endif /* shm_hashtable_h */
