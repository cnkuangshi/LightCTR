//
//  ring_collect.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/7.
//  Copyright © 2017 SongKuangshi. All rights reserved.
//

#ifndef ring_collect_h
#define ring_collect_h

#include "dist_machine_abst.h"
#include <vector>
#include "../common/buffer_fusion.h"
#include "../common/avx.h"
#include "../common/barrier.h"
#include "../common/lock.h"

const time_t kTimeoutRetryMSInterval = 10000;

// especially design for GPUs' collective ring-reduce
template<typename T>
class Worker_RingReduce : public Dist_Machine_Abst {
public:
    explicit Worker_RingReduce(size_t ring_size) : _ring_size(ring_size) {
        
        cur_node_id = Worker_RingReduce::Rank(); // begin from 0
        assert(cur_node_id >= 0);
        recv_from_id = BEGIN_ID_OF_WORKER + 1 + (cur_node_id + _ring_size - 1) % _ring_size;
        send_to_id = BEGIN_ID_OF_WORKER + 1 + (cur_node_id + 1 + _ring_size) % _ring_size;
        
        printf("[RING] Ring %zu -> %zu -> %zu\n", recv_from_id,
                    BEGIN_ID_OF_WORKER + 1 + cur_node_id, send_to_id);
        // check router
        gDelivery.get_router(recv_from_id);
        gDelivery.get_router(send_to_id);
        // TODO recovering boot mode, Copy parameters from the other conventional Ring worker
        // and Start send and receive by processing epoch_version
    }
    
    ~Worker_RingReduce() {
        assert(cache.empty());
        segment_size_arr.clear();
        segment_end_arr.clear();
    }
    
    void syncGradient(std::shared_ptr<BufferFusion<T> > _buf_fusion,
                      size_t epoch,
                      bool do_Average = true) {
        init(_buf_fusion);
        step_version = epoch * (2 * _ring_size - 2);
        
        // Firstly do all-reduce
        reduce_step(_buf_fusion);
        
        // Secondly do all-gather
        gather_step(_buf_fusion);
        
        // Finally
        if (likely(do_Average)) {
            const float scalar = 1.0 / _ring_size;
            _buf_fusion->transform(0,
                                  _buf_fusion->size(),
                                  [scalar](T* begin, T* end) {
                                      avx_vecScale(begin, begin, end - begin, scalar);
                                  });
        }
#ifdef DEBUG
        printf("[RING] **** Epoch %zu synchronizer completed ****\n", epoch);
#endif
    }
    
    void syncInitializer(std::shared_ptr<BufferFusion<T> > _buf_fusion) {
        init(_buf_fusion);
        step_version = _ring_size - 1;
        
        gather_step(_buf_fusion);
    }
    
    inline size_t Rank() const { // Rank begin from 0
        return Dist_Machine_Abst::Rank() - 1;
    }
    
private:
    void init(std::shared_ptr<BufferFusion<T> > _buf_fusion) {
        assert(_ring_size > 0);
        if (unlikely(_param_size != _buf_fusion->size())) {
        
            _param_size = _buf_fusion->size();
            assert(_param_size > 0);
            
            segment_size_arr.resize(_ring_size);
            segment_end_arr.resize(_ring_size);
            const size_t seg_size = _param_size / _ring_size;
            const size_t seg_res = _param_size % _ring_size;
            
            for (size_t i = 0; i < _ring_size; i++) {
                segment_size_arr[i] = seg_size;
                if (i < seg_res) {
                    segment_size_arr[i]++;
                }
                if (i == 0) {
                    segment_end_arr[0] = segment_size_arr[0];
                } else {
                    segment_end_arr[i] = segment_end_arr[i - 1] + segment_size_arr[i];
                }
            }
        }
        
        regist_reduce_gather_handler(_buf_fusion);
    }
    
    void reduce_step(std::shared_ptr<BufferFusion<T> > _buf_fusion) {
        for (size_t i = 0; i < _ring_size - 1; i++) {
            const size_t send_segment_id = (cur_node_id + _ring_size - i) % _ring_size;
            recv_segment_id = (cur_node_id + _ring_size - i - 1) % _ring_size;
            
            // send segment to next-skip on the ring topology
            rcv_offset = segment_end_arr[recv_segment_id] - segment_size_arr[recv_segment_id];
            step_version++;
            wmb();
            
            step_barrier.reset();
            
            // receive segment from last-skip on the ring topology
            {
                unique_lock<SpinLock> glock(cache_lock);
                if (!cache.empty()) {
#ifdef DEBUG
                    printf("[REDUCE] step = %zu read from cache\n", step_version);
#endif
                    auto request = cache.front();
                    cache.pop();
                    assert(request->epoch_version == step_version);
                    _do_reduce(_buf_fusion, request->content);
                    step_barrier.unblock();
                }
            }
            
            PackageDescript desc(REQUEST_PUSH, step_version);
            Buffer* buffer_ptr = nullptr;
            _buf_fusion->memcpy_out(&buffer_ptr,
                              segment_end_arr[send_segment_id] - segment_size_arr[send_segment_id],
                              segment_size_arr[send_segment_id]);
            desc.content = std::move(*buffer_ptr);
#ifdef DEBUG
            desc.callback = [this](std::shared_ptr<PackageDescript> resp_package) {
                printf("[REDUCE] send step = %zu package success\n", step_version);
//                assert(resp_package->epoch_version && resp_package->epoch_version <= step_version);
            };
#endif
            bool send_status = false;
            do {
                send_status = gDelivery.send_sync(desc, send_to_id, kTimeoutRetryMSInterval);
#ifdef DEBUG
                if (unlikely(!send_status)) {
                    // TODO dynamic waiting interval for network delay or crash of some machines
                    printf("[ERROR][REDUCE] send step = %zu package failed, retry\n", step_version);
                }
#endif
            } while(!send_status);
            
            step_barrier.block();
        }
    }
    
    void gather_step(std::shared_ptr<BufferFusion<T> > _buf_fusion) {
        for (size_t i = 0; i < _ring_size - 1; i++) {
            const size_t send_segment_id = (cur_node_id + 1 + _ring_size - i) % _ring_size;
            recv_segment_id = (cur_node_id + _ring_size - i) % _ring_size;
            
            // send segment to next-skip on the ring topology
            rcv_offset = segment_end_arr[recv_segment_id] - segment_size_arr[recv_segment_id];
            step_version++;
            wmb();
            
            step_barrier.reset();
            
            // receive segment from last-skip on the ring topology
            {
                unique_lock<SpinLock> glock(cache_lock);
                if (!cache.empty()) {
#ifdef DEBUG
                    printf("[GATHER] step = %zu read from cache\n", step_version);
#endif
                    auto request = cache.front();
                    cache.pop();
                    assert(request->epoch_version == step_version);
                    _do_gather(_buf_fusion, request->content);
                    step_barrier.unblock();
                }
            }
            
            PackageDescript desc(REQUEST_PUSH, step_version);
            Buffer* buffer_ptr = nullptr;
            _buf_fusion->memcpy_out(&buffer_ptr,
                                    segment_end_arr[send_segment_id] - segment_size_arr[send_segment_id],
                                    segment_size_arr[send_segment_id]);
            desc.content = std::move(*buffer_ptr);
#ifdef DEBUG
            desc.callback = [this](std::shared_ptr<PackageDescript> resp_package) {
                printf("[GATHER] send step = %zu package success\n", step_version);
//                assert(resp_package->epoch_version && resp_package->epoch_version <= step_version);
            };
#endif
            
            bool send_status = false;
            do {
                send_status = gDelivery.send_sync(desc, send_to_id, kTimeoutRetryMSInterval);
#ifdef DEBUG
                if (unlikely(!send_status)) {
                    printf("[ERROR][GATHER] send step %zu package failed, retry\n", step_version);
                }
#endif
            } while(!send_status);
            
            step_barrier.block();
        }
    }
    
    void _do_reduce(std::shared_ptr<BufferFusion<T> > _buf_fusion, Buffer& data) {
        assert(segment_size_arr[recv_segment_id] * sizeof(T) == data.size());
        
        // accumulate gradients
        if (typeid(T) == typeid(float)) { // try to use AVX
            const float* buffer = reinterpret_cast<const float*>(data.buffer());
            
            _buf_fusion->transform(rcv_offset,
                                  segment_size_arr[recv_segment_id],
                                  [&buffer](T* begin, T* end) {
                                      avx_vecAdd(buffer, begin, begin, end - begin);
                                      buffer += end - begin;
                                  });
        } else {
            
            _buf_fusion->transform(rcv_offset,
                                  segment_size_arr[recv_segment_id],
                                  [&data](T* begin, T* end) {
                                      T grad_value;
                                      for (size_t i = 0; i < end - begin; i++) {
                                          data >> grad_value;
                                          *(begin + i) += grad_value;
                                      }
                                  });
            assert(data.readEOF());
        }
    }
    
    void _do_gather(std::shared_ptr<BufferFusion<T> > _buf_fusion, const Buffer& data) {
        assert(segment_size_arr[recv_segment_id] * sizeof(T) == data.size());
        
        const float* buffer = reinterpret_cast<const float*>(data.buffer());
        _buf_fusion->memcpy_in(rcv_offset, buffer, segment_size_arr[recv_segment_id]);
    }
    
    void regist_reduce_gather_handler(std::shared_ptr<BufferFusion<T> > _buf_fusion) {
        request_handler_t handler = [this, _buf_fusion](std::shared_ptr<PackageDescript> request,
                                           PackageDescript& response) {
            rmb();
            assert(request->node_id > BEGIN_ID_OF_WORKER);
            const size_t worker_id = request->node_id;
            assert(worker_id == recv_from_id);
            
//            assert(request->epoch_version >= step_version);
            {
                unique_lock<SpinLock> glock(cache_lock);
                if (step_version != request->epoch_version) {
                    // cache the request into deque and response the situation
                    cache.push(request);
#ifdef DEBUG
                    printf("[RING] receive not match %zu expected %zu, cache it\n",
                           request->epoch_version, step_version);
#endif
                    response.epoch_version = step_version;
                    return;
                }
            }
            
            assert(request->content.size() % sizeof(T) == 0);
#ifdef DEBUG
            printf("[RING] step %zu: recv %zu gradients\n",
                   step_version, request->content.size() / sizeof(T));
#endif
            
            const size_t type = step_version % (2 * _ring_size - 2);
            if (type > 0 && type < _ring_size) {
                _do_reduce(_buf_fusion, request->content);
            } else {
                _do_gather(_buf_fusion, request->content);
            }
            
            response.epoch_version = step_version;
            step_barrier.unblock();
        };
        
        gDelivery.regist_handler(REQUEST_PUSH, std::move(handler));
    }
    
    std::vector<size_t> segment_size_arr;
    std::vector<size_t> segment_end_arr;
    
    volatile size_t step_version;
    volatile size_t rcv_offset;
    volatile size_t recv_segment_id;
    
    queue<std::shared_ptr<PackageDescript> > cache;
    SpinLock cache_lock;
    
    size_t _param_size;
    const size_t _ring_size = 0;
    
    size_t cur_node_id;
    size_t recv_from_id;
    size_t send_to_id;
    
    Barrier step_barrier;
};

#endif /* ring_collect_h */
