//
//  consistent_hash.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/6.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef consistent_hash_h
#define consistent_hash_h

#include "../common/hash.h"
#include <cstring>
#include <sstream>
#include <map>

// Make data shardings ditributed in PS clusters by DHT
class ConsistentHash {
public:
    static ConsistentHash& Instance() { // singleton
        static std::once_flag once;
        static ConsistentHash consist;
        std::call_once(once, [] {
            assert(__global_cluster_ps_cnt > 0);
            consist.init(__global_cluster_ps_cnt);
        });
        return consist;
    }
    
    template <typename TKey>
    inline uint32_t getNode(TKey key) {
        uint32_t partition = murMurHash(key);
        std::map<uint32_t, uint32_t>::iterator it =
            server_nodes.lower_bound(partition);
        
        if(it == server_nodes.end()) {
            return server_nodes.begin()->second;
        }
        return it->second;
    }
    
private:
    ConsistentHash() {
        
    }
    ConsistentHash(const ConsistentHash&) = delete;
    ConsistentHash(ConsistentHash&&) = delete;
    ConsistentHash &operator=(const ConsistentHash &) = delete;
    ConsistentHash &operator=(ConsistentHash &&) = delete;
    
    void init(uint32_t _node_cnt) {
        node_cnt = _node_cnt;
        for (uint32_t i = 0; i < node_cnt; i++) {
            for (uint32_t j = 0; j < virtual_node_cnt; j++) {
                std::stringstream node_key;
                node_key << i << "-" << j;
                uint32_t partition = murMurHash(node_key.str());
                server_nodes[partition] = i;
            }
        }
    }
    
    uint32_t node_cnt;
    const uint32_t virtual_node_cnt{5}; // num of Replicas
    
    std::map<uint32_t, uint32_t> server_nodes;
};

#endif /* consistent_hash_h */
