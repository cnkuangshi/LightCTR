//
//  paramserver.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef paramserver_h
#define paramserver_h

#include "../common/network.h"
#include "../common/barrier.h"
#include "../common/lock.h"
#include <unordered_map>
#include "../util/gradientUpdater.h"

// provide pull and push of parameters shardings to workers
template <typename TKey, typename TValue>
class ParamServer {
    struct ValueWrapper {
        TValue data;
        TValue data_accum; // reserve for adagrad
    };
public:
    ParamServer() : gDelivery(Delivery::Instance()) {
        gDelivery.set_node_id(BEGIN_ID_OF_PS);
        regist_curNode_toMaster();
        regist_ack_handler();
        regist_fin_handler();
        
        serving_barrier.block();
        status_serving = true;
        
        regist_pull_push_handler();
        
        terminate_barrier.block();
    }
    
    inline size_t Rank() const { // PS Rank begin from 1
        assert(status_serving);
        return gDelivery.node_id();
    }
    inline bool status() const {
        return status_serving;
    }
    
private:
    void regist_curNode_toMaster() {
        PackageDescript desc(REQUEST_HANDSHAKE);
        std::string local_addr_str = gDelivery.local_addr().toString();
        desc.content.append(local_addr_str.c_str(), local_addr_str.length());
        
        desc.callback = [this](std::shared_ptr<PackageDescript> resp_package) {
            size_t node_id;
            resp_package->content >> node_id;
            printf("[PS] Complete Register cur_node_id = %zu\n", node_id);
            gDelivery.set_node_id(node_id);
            assert(gDelivery.node_id() >= BEGIN_ID_OF_PS);
        };
        gDelivery.send_sync(std::move(desc), 0);
    }
    
    void regist_ack_handler() {
        request_handler_t ack_handler = [this](
                                               std::shared_ptr<PackageDescript> request,
                                               PackageDescript& response) {
            size_t w_id = 1 + BEGIN_ID_OF_WORKER;
            while (!request->content.readEOF()) { // read keys needed by worker
                Addr w_addr(request->content);
                printf("[PS] add worker_id = %zu router\n", w_id);
                gDelivery.regist_router(w_id++, w_addr);
            }
            serving_barrier.unblock();
        };
        gDelivery.regist_handler(REQUEST_ACK, std::move(ack_handler));
    }
    
    void regist_fin_handler() {
        request_handler_t fin_handler = [this](
                                               std::shared_ptr<PackageDescript> request,
                                               PackageDescript& response) {
            gDelivery.shutdown();
            terminate_barrier.unblock();
        };
        gDelivery.regist_handler(REQUEST_FIN, std::move(fin_handler));
    }
    
    void regist_pull_push_handler() {
        request_handler_t pull_handler = [this](
                                             std::shared_ptr<PackageDescript> request,
                                             PackageDescript& response) {
            rwlock.rlock();
            
            TKey key;
            assert(request->content.size() % sizeof(key) == 0);
            printf("[PS PULL] recv %zu keys\n", request->content.size() / sizeof(key));
            
            while (!request->content.readEOF()) { // read keys needed by worker
                request->content >> key;
                
                auto it = paramShardTable.find(key);
                if (it == paramShardTable.end()) {
                    rwlock.unlock();
                    {
                        rwlock.wlock();
                        it = check_and_find(key, true);
                        rwlock.unlock();
                    }
                    rwlock.rlock();
                }
                assert(it->second.data.checkValid());
                // return pull target param by pair
                auto pair = make_pair(it->first, it->second.data);
                response.content << pair;
            }
            rwlock.unlock();
        };
        request_handler_t push_handler = [this](
                                             std::shared_ptr<PackageDescript> request,
                                             PackageDescript& response) {
            rwlock.wlock();
            
            std::pair<TKey, TValue> data_pair;
            assert(request->content.size() % sizeof(data_pair) == 0);
            printf("[PS PUSH] recv %zu pairs\n", request->content.size() / sizeof(data_pair));
            
            while (!request->content.readEOF()) {
                request->content >> data_pair;
                assert(data_pair.second.checkValid());
                
                auto it = check_and_find(data_pair.first);
                // apply grad into local param
                // simple SGD
//                it->second.data - data_pair.second /
//                                    ((double)GradientUpdater::__global_minibatch_size
//                                        / GradientUpdater::__global_learning_rate);
                // adagrad
                TValue grad = data_pair.second / GradientUpdater::__global_minibatch_size;
                it->second.data_accum + grad * grad;
                it->second.data - data_pair.second /
                    (it->second.data_accum.sqrt() / GradientUpdater::__global_learning_rate);
                
                assert(it->second.data.checkValid());
                assert(it->second.data_accum.checkValid());
            }
            rwlock.unlock();
            // TODO params backup checkpoint to Hard Disk periodicity
        };
        gDelivery.regist_handler(REQUEST_PULL, std::move(pull_handler));
        gDelivery.regist_handler(REQUEST_PUSH, std::move(push_handler));
    }
    
    typename std::unordered_map<TKey, ValueWrapper>::iterator check_and_find(
                                               TKey key,
                                               bool just_init = false) {
        auto it = just_init ? paramShardTable.end() : paramShardTable.find(key);
        if (it == paramShardTable.end()) { // first time push, do param init
            std::pair<TKey, ValueWrapper> init_data_pair;
            init_data_pair.first = key;
            ValueWrapper val_wrapper;
            val_wrapper.data.initParam();
            val_wrapper.data_accum = TValue(1e-12);
            init_data_pair.second = std::move(val_wrapper);
            auto it_pos = paramShardTable.insert(std::move(init_data_pair));
            if (!it_pos.second) { // another write priority
                it = paramShardTable.find(key);
                assert(it != paramShardTable.end());
            } else {
                it = it_pos.first;
            }
        }
        return it;
    }
    
    std::unordered_map<TKey, ValueWrapper> paramShardTable;
    RWLock rwlock;
    
    bool status_serving{false};
    Barrier serving_barrier;
    Barrier terminate_barrier;
    Delivery&& gDelivery;
};

#endif /* paramserver_h */
