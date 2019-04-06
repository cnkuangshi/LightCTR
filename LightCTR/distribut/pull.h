//
//  pull.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef pull_h
#define pull_h

#include <unordered_map>
#include <map>
#include <vector>
#include <atomic>
#include "consistent_hash.h"
#include "../common/thread_pool.h"
#include "../common/barrier.h"
#include "../common/network.h"
#include "../common/buffer_fusion.h"
#include "../util/matrix.h"

// Pull params from PS
class Pull {
    
public:
    Pull() = delete;
    explicit Pull(char _headByte) :
             headByte(_headByte),
             gDelivery(Delivery::Instance()),
             gConsistentHash(ConsistentHash::Instance()) {
    }
    
    void registTensorFusion(std::shared_ptr<BufferFusion<float> > _buf_fusion) {
        assert(headByte == 'T');
        buf_fusion = _buf_fusion;
    }
    
    // pull params used keys
    // when headByte == 'N' means sparse vector, unordered_map<fid, float value>
    // when headByte == 'T' means tensor vector, unordered_map<fid, offset>
    template <class TKey, class TValue>
    void sync(std::unordered_map<TKey, TValue> &keys, size_t epoch) {
        if (headByte == 'T')
            assert(buf_fusion);
        assert(epoch > 0);
        int candidate_ps = 0;
        
        size_t recv_param_cnt = 0;
        do {
            Barrier barrier;
            recv_param_cnt = 0;
            sendToPS(keys, candidate_ps, epoch,
                     [&barrier, &candidate_ps, &recv_param_cnt](size_t inc) {
                         recv_param_cnt += inc;
                         candidate_ps--;
                         assert(candidate_ps >= 0);
                         if (candidate_ps <= 0) {
                             barrier.unblock();
                         }
                     });
            barrier.block();
            if (recv_param_cnt != keys.size()) {
                puts("[WORKER PULL] Wait for other workers based SSP");
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        } while(recv_param_cnt != keys.size());
    }
    
private:
    template <class TKey, class TValue>
    void sendToPS(std::unordered_map<TKey, TValue> &keys,
                  int& candidate_ps,
                  size_t epoch,
                  std::function<void(size_t)> callback) {
        std::map<size_t, std::vector<TKey> > pull_map;
        
        for (auto it = keys.begin(); it != keys.end(); it++) {
            const size_t to_id = BEGIN_ID_OF_PS +
                                 gConsistentHash.getNode(it->first);
            if (pull_map.count(to_id) == 0) {
                pull_map[to_id] = std::vector<TKey>();
                candidate_ps++;
            }
            pull_map[to_id].emplace_back(std::move(it->first));
        }
        
        for (auto &item : pull_map) {
            const size_t to_id = item.first;
            PackageDescript desc(REQUEST_PULL, epoch);
            desc.content << headByte;
            
            _pack_req(desc, keys, item.second, callback);
            gDelivery.send_async(desc, to_id);
        }
        
#ifdef DEBUG
        printf("[WORKER Pull] %zu %c Keys Sended\n", keys.size(), headByte);
#endif
    }
    
    template <class TKey, class TValue>
    void _pack_req(PackageDescript &desc,
                   std::unordered_map<TKey, TValue> &keys,
                   const std::vector<TKey> &keys_on_ps,
                   std::function<void(size_t)> callback) {
        for (auto &key : keys_on_ps) {
            // pull VarUint keys
            desc.content.appendVarUint(key);
        }
        desc.callback = [&keys, callback](std::shared_ptr<PackageDescript> resp_package) {
            std::pair<TKey, TValue> data_pair;
            
            size_t inc = 0;
            while(!resp_package->content.readEOF()) {
                // parsing pull response by VarUint & float16_t
                resp_package->content.readVarUint(&data_pair.first);
                resp_package->content.readHalfFloat(&data_pair.second);
                
                assert(data_pair.second.checkValid());
                
                auto it = keys.find(data_pair.first);
                assert(it != keys.end());
                
                it->second = data_pair.second;
                inc++;
            }
            assert(resp_package->content.readEOF());
            
            if (callback) {
                callback(inc);
            }
        };
    }
    
    template <class TKey>
    void _pack_req(PackageDescript &desc,
                   std::unordered_map<TKey, size_t> &keys,
                   const std::vector<TKey> &keys_on_ps,
                   std::function<void(size_t)> callback) {
        for (auto &key : keys_on_ps) {
            // pull VarUint keys
            desc.content.appendVarUint(key);
            
            auto memAddr = buf_fusion->getMemory(keys[key]);
            desc.content.appendVarUint(memAddr.second);
        }
        desc.callback = [this, &keys, callback](
                                                std::shared_ptr<PackageDescript> resp_package) {
            std::pair<TKey, float> data_pair;
            TKey length;
            
            size_t inc = 0;
            while(!resp_package->content.readEOF()) {
                resp_package->content.readVarUint(&data_pair.first);
                resp_package->content.readVarUint(&length);
                
                size_t offset = keys[data_pair.first];
                auto memAddr = buf_fusion->getMemory(offset);
                assert(memAddr.second == length);
                
                for (size_t i = 0; i < length; i++) {
                    resp_package->content.readHalfFloat(&data_pair.second);
                    *(memAddr.first + i) = data_pair.second;
                }
                
                inc++;
            }
            assert(resp_package->content.readEOF());
            
            if (callback) {
                callback(inc);
            }
        };
    }
    
    char headByte = 'N';
    std::shared_ptr<BufferFusion<float> > buf_fusion = nullptr;
    
    Delivery& gDelivery;
    ConsistentHash& gConsistentHash;
};

#endif /* pull_h */
