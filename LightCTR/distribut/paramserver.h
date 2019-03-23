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
#include "../common/avx.h"
#include <unordered_map>
#include "../util/gradientUpdater.h"
#include "dist_machine_abst.h"

const size_t kStalenessStepThreshold = 10;

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
    };
    struct TensorWrapper {
        TensorWrapper(size_t _len) {
            data.resize(_len);
            std::fill(data.begin(), data.end(), 0);
        }
        vector<float> data;
    };
public:
    ParamServer(UpdaterType _updaterType = UpdaterType::SGD) :
    gDelivery(Delivery::Instance()), updaterType(_updaterType) {
        gDelivery.set_node_id(BEGIN_ID_OF_PS);
        regist_curNode_toMaster();
        regist_ack_handler();
        regist_fin_handler();
        
        serving_barrier.block();
        status_serving = true;
        
        paramShardTable.reserve(1 << 20); // reserve 1000k
        paramShardTable.rehash(1 << 20); // prevent rehashing of unordered_map
        
        tensorShardTable.reserve(1 << 20);
        tensorShardTable.rehash(1 << 20);
        
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
            serving_barrier.unblock();
        };
        gDelivery.send_async(desc, 0);
    }
    
    void regist_ack_handler() {
        request_handler_t ack_handler = [this](
                                               std::shared_ptr<PackageDescript> request,
                                               PackageDescript& response) {
            size_t w_id = 1 + BEGIN_ID_OF_WORKER;
            while (!request->content.readEOF()) { // read keys needed by worker
                Addr w_addr(request->content);
                printf("[PS] add worker_id = %zu router\n", w_id);
                gDelivery.regist_router(w_id++, std::move(w_addr));
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
            {
                std::unique_lock<std::mutex> lock(step_lock);
                
                if (request->epoch_version > last_epoch_version &&
                    // TODO dynamic control waiting kStalenessStepThreshold
                    staleness_epoch_version > kStalenessStepThreshold) {
                    // None values response indicate that worker should wait a moment
                    printf("[PS PULL] staleness %zu but recv %zu\n",
                           staleness_epoch_version, request->epoch_version);
                    return;
                }
            }
            // Lock-free pulling based by Hogwild!
            TKey key, length;
            char headByte;
            request->content >> headByte;
            assert(headByte == 'N' || headByte == 'T');
            
            while (!request->content.readEOF()) { // read keys needed by worker
                request->content.readVarUint(&key);
                if (headByte == 'T') {
                    request->content.readVarUint(&length);
                    auto it = tensorShardTable.find(key);
                    if (it == tensorShardTable.end()) {
                        auto it_pos = tensorShardTable.insert(std::make_pair(key, TensorWrapper(length)));
                        if (!it_pos.second) { // another write priority
                            it = tensorShardTable.find(key);
                            assert(it != tensorShardTable.end());
                        } else {
                            it = it_pos.first;
                        }
                    }
                    assert(length == it->second.data.size());
                    response.content.appendVarUint(key);
                    response.content.appendVarUint(length);
                    for (size_t i = 0; i < length; i++) {
                        response.content << Float16(&it->second.data[i]).float16_value();
                    }
                    continue;
                }
                
                auto it = paramShardTable.find(key);
                if (it == paramShardTable.end()) {
                    // should double check by check_and_find
                    it = check_and_find(key);
                }
                assert(it->second.data_readonly.checkValid());
                
                // return pull target param by pair
                response.content.appendVarUint(key);
                response.content << Float16(&it->second.data_readonly).float16_value();
            }
            assert(request->content.readEOF());
        };
        
        request_handler_t push_handler = [this](
                                             std::shared_ptr<PackageDescript> request,
                                             PackageDescript& response) {
            std::pair<TKey, TValue> data_pair;
            assert(request->node_id >= BEGIN_ID_OF_WORKER);
            const size_t worker_id = request->node_id - BEGIN_ID_OF_WORKER - 1;
            assert(worker_id < __global_cluster_worker_cnt);
            
            {
                std::unique_lock<std::mutex> lock(step_lock);
                
                if (staleness_epoch_version > 0 &&
                    worker_id == staleness_workerid &&
                    staleness_epoch_version > (int)last_epoch_version - (int)request->epoch_version) {
                    // slowest node is catching up
                    staleness_epoch_version = std::max(0,
                                                       (int)last_epoch_version - (int)request->epoch_version);
                }
                if (staleness_epoch_version < (int)last_epoch_version - (int)request->epoch_version) {
                    staleness_epoch_version = std::max(0,
                                                       (int)last_epoch_version - (int)request->epoch_version);
                    staleness_workerid = worker_id;
                }
                if (request->epoch_version + kStalenessStepThreshold < last_epoch_version) {
                    printf("[PS PUSH] last version %zu but recv %zu, drop behindhand\n",
                           last_epoch_version, request->epoch_version);
                    return;
                }
                last_epoch_version = std::max(last_epoch_version, request->epoch_version);
            }
            
            TKey length;
            char headByte;
            request->content >> headByte;
            assert(headByte == 'N' || headByte == 'T');
            
            while (!request->content.readEOF()) {
                request->content.readVarUint(&data_pair.first);
                
                if (headByte == 'T') {
                    request->content.readVarUint(&length);
                    
                    std::vector<float> values;
                    for (size_t i = 0; i < length; i++) {
                        request->content.readHalfFloat(&data_pair.second);
                        values.push_back(data_pair.second.w);
                    }
                    
                    auto it = tensorShardTable.find(data_pair.first);
                    assert(length == it->second.data.size());
                    
                    // simple SGD
                    float scaler = - 1.0 * GradientUpdater::__global_learning_rate
                                    / GradientUpdater::__global_minibatch_size;
                    avx_vecScale(values.data(), values.data(), length, scaler);
                    avx_vecAdd(it->second.data.data(), values.data(),
                               it->second.data.data(), length);
                    
                    continue;
                }
                
                request->content.readHalfFloat(&data_pair.second);
                
                assert(data_pair.second.checkValid());
                
                // do gradient clipping and rescale
                data_pair.second * rescaleGrad;
                
                auto it = paramShardTable.find(data_pair.first);
                
                // apply grad into local param
                if (updaterType == UpdaterType::DCASGD) {
                    // delayed compensation asynchronous SGD
                    const float dcasgd_lambda = 0.1;
                    
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
                    // delayed compensation asynchronous SGD adaptive
                    const float dcasgd_lambda = 0.1;
                    const float momentum_rate = 0.95;
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
                    ((float)GradientUpdater::__global_minibatch_size
                     / GradientUpdater::__global_learning_rate);
                }
                // at last swap data and data_readonly
                it->second.data_readonly = it->second.data;
                
                assert(it->second.data.checkValid());
                assert(it->second.data_accum.checkValid());
            }
            
            assert(request->content.readEOF());
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
    
    const float rescaleGrad = 1.0f;
    
    std::unordered_map<TKey, ValueWrapper> paramShardTable;
    std::unordered_map<TKey, TensorWrapper> tensorShardTable;
    std::mutex step_lock;
    size_t last_epoch_version{1};
    size_t staleness_epoch_version{0};
    size_t staleness_workerid{0};
    
    UpdaterType updaterType;
    bool status_serving{false};
    Barrier serving_barrier{2};
    Barrier terminate_barrier;
    Delivery& gDelivery;
};

#endif /* paramserver_h */
