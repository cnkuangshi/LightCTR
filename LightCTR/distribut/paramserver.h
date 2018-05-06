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

enum UpdaterType {
    SGD = 0,
    Adagrad,
    DCASGD,
    DCASGDA
};
#define UPDATER_TYPE 1

// provide pull and push of parameters shardings to workers
template <typename TKey, typename TValue>
class ParamServer {
    struct ValueWrapper {
        TValue data;
        TValue data_readonly;
        TValue data_accum; // reserve for adagrad
        // reserve for DCASGD to cache each worker's latest version of parameter
        TValue* shadow_copies;
        time_t lastUpdateTime;
    };
public:
    ParamServer(UpdaterType _updaterType = UpdaterType::DCASGDA) :
    gDelivery(Delivery::Instance()), updaterType(_updaterType) {
        gDelivery.set_node_id(BEGIN_ID_OF_PS);
        regist_curNode_toMaster();
        regist_ack_handler();
        regist_fin_handler();
        
        serving_barrier.block();
        
        paramShardTable.reserve(1 << 20); // reserve 1000k
        paramShardTable.rehash(1 << 20); // prevent rehashing of unordered_map
        
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
            // Lock-free pulling based by Hogwild!
            TKey key;
            assert(request->content.size() % sizeof(key) == 0);
            printf("[PS PULL] recv %zu keys\n", request->content.size() / sizeof(key));
            
            size_t cnt = 0, skip_cnt = 0;
            while (!request->content.readEOF()) { // read keys needed by worker
                request->content >> key;
                
                auto it = paramShardTable.find(key);
                if (it == paramShardTable.end()) {
                    rwlock.wlock(); // should double check by check_and_find
                    it = check_and_find(key);
                    rwlock.unlock();
                }
                assert(it->second.data_readonly.checkValid());
                if (!it->second.data_readonly.checkPreferredValue()) {
                    // select preferred feature to reduce transmission cost
                    skip_cnt++;
                    continue;
                }
                // return pull target param by pair
                auto pair = make_pair(it->first, it->second.data_readonly);
                response.content << pair;
                cnt++;
            }
            
            printf("[PS PULL] send %zu pairs, skip %zu pairs\n", cnt, skip_cnt);
        };
        
        request_handler_t push_handler = [this](
                                             std::shared_ptr<PackageDescript> request,
                                             PackageDescript& response) {
            std::pair<TKey, TValue> data_pair;
            assert(request->node_id >= BEGIN_ID_OF_WORKER);
            const size_t worker_id = request->node_id - BEGIN_ID_OF_WORKER - 1;
            assert(worker_id < __global_cluster_worker_cnt);
            assert(request->content.size() % sizeof(data_pair) == 0);
            printf("[PS PUSH] recv %zu pairs\n", request->content.size() / sizeof(data_pair));
            
            rwlock.wlock();
            
            size_t saved_cnt = 0, skip_cnt = 0;
            while (!request->content.readEOF()) {
                request->content >> data_pair;
                assert(data_pair.second.checkValid());
                
                // do gradient clipping and rescale
                data_pair.second * rescaleGrad;
                if (!data_pair.second.checkPreferredValue()) {
                    skip_cnt++;
                    continue;
                }
                
                auto it = paramShardTable.find(data_pair.first);
                
                // apply grad into local param
                if (updaterType == UpdaterType::DCASGD) {
                    // delayed compensation asynchronous SGD
                    const double dcasgd_lambda = 0.1;
                    
                    TValue grad = data_pair.second / GradientUpdater::__global_minibatch_size;
                    assert(grad.checkValid());
                    TValue curValue(it->second.data);
                    assert(it->second.shadow_copies[worker_id].checkValid());
                    TValue reserveGrad(grad);
                    
                    reserveGrad + ((grad * grad)
                        * (curValue - it->second.shadow_copies[worker_id])
                        * dcasgd_lambda);
                    assert(reserveGrad.checkValid());
                    it->second.data - (reserveGrad * GradientUpdater::__global_learning_rate);
                    it->second.shadow_copies[worker_id] = it->second.data;
                } else if (updaterType == UpdaterType::DCASGDA) {
                    // delayed compensation asynchronous SGD
                    const double dcasgd_lambda = 0.1;
                    const double momentum_rate = 0.95;
                    TValue grad = data_pair.second / GradientUpdater::__global_minibatch_size;
                    it->second.data_accum * momentum_rate + ((grad * grad) * (1 - momentum_rate));
                    
                    assert(it->second.shadow_copies[worker_id].checkValid());
                    TValue curValue(it->second.data);
                    TValue reserveGrad(grad);
                    TValue sqrtValue;
                    it->second.data_accum.sqrt(sqrtValue);
                    reserveGrad + ((grad * grad)
                        * (curValue - it->second.shadow_copies[worker_id])
                        * dcasgd_lambda
                        / sqrtValue);
                    
                    it->second.data - (reserveGrad * GradientUpdater::__global_learning_rate);
                    it->second.shadow_copies[worker_id] = it->second.data;
                } else if (updaterType == UpdaterType::Adagrad) {
                    // adagrad
                    TValue grad = data_pair.second / GradientUpdater::__global_minibatch_size;
                    it->second.data_accum + grad * grad;
                    TValue sqrtValue;
                    it->second.data_accum.sqrt(sqrtValue);
                    it->second.data - data_pair.second /
                        (sqrtValue / GradientUpdater::__global_learning_rate);
                } else {
                    // simple SGD
                    it->second.data - data_pair.second /
                    ((double)GradientUpdater::__global_minibatch_size
                     / GradientUpdater::__global_learning_rate);
                }
                // at last swap data and data_readonly
                it->second.data_readonly = it->second.data;
                it->second.lastUpdateTime = time(NULL);
                saved_cnt++;
                
                assert(it->second.data.checkValid());
                assert(it->second.data_accum.checkValid());
            }
            
            rwlock.unlock();
            
            printf("[PS PUSH] saved %zu pairs, skip %zu pairs\n", saved_cnt, skip_cnt);
            // TODO params backup checkpoint to Hard Disk periodicity
        };
        gDelivery.regist_handler(REQUEST_PULL, std::move(pull_handler));
        gDelivery.regist_handler(REQUEST_PUSH, std::move(push_handler));
    }
    
    typename std::unordered_map<TKey, ValueWrapper>::iterator check_and_find(
                                               TKey key) {
        auto it = paramShardTable.find(key);
        if (it == paramShardTable.end()) { // first time push, do param init
            std::pair<TKey, ValueWrapper> init_data_pair;
            init_data_pair.first = key;
            ValueWrapper val_wrapper;
            val_wrapper.data.initParam();
            val_wrapper.data_accum = TValue(1e-8);
            val_wrapper.data_readonly = val_wrapper.data;
            val_wrapper.shadow_copies = NULL;
            val_wrapper.lastUpdateTime = time(NULL);
            if (updaterType == UpdaterType::DCASGD || updaterType == UpdaterType::DCASGDA) {
                val_wrapper.shadow_copies = new TValue[__global_cluster_worker_cnt]();
            }
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
    
    const double rescaleGrad = 1.0f;
    
    std::unordered_map<TKey, ValueWrapper> paramShardTable;
    RWLock rwlock;
    
    UpdaterType updaterType;
    bool status_serving{false};
    Barrier serving_barrier;
    Barrier terminate_barrier;
    Delivery&& gDelivery;
};

#endif /* paramserver_h */
