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
#include "../common/avx.h"

// especially design for GPUs' collective ring-reduce
template<typename T>
class Worker_RingReduce : public Dist_Machine_Abst {
public:
    Worker_RingReduce(size_t param_size, size_t ring_size)
                    : _param_size(param_size), _ring_size(ring_size) {
        assert(param_size > 0 && ring_size > 0);
        assert(ring_size == __global_cluster_worker_cnt);
                        
        segment_size_arr.resize(ring_size);
        segment_end_arr.resize(ring_size);
        const size_t seg_size = param_size / ring_size;
        const size_t seg_res = param_size % ring_size;
        
        for (size_t i = 0; i < ring_size; i++) {
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
        cur_node_id = Rank() - 1; // begin from 0
        assert(cur_node_id >= 0);
        recv_from_id = BEGIN_ID_OF_WORKER + 1 + (cur_node_id + _ring_size - 1) % _ring_size;
        send_to_id = BEGIN_ID_OF_WORKER + 1 + (cur_node_id + 1 + _ring_size) % _ring_size;
        
        printf("[RING] Ring %zu -> %zu -> %zu\n", recv_from_id,
                    BEGIN_ID_OF_WORKER + 1 + cur_node_id, send_to_id);
        gDelivery.get_router(recv_from_id);
        gDelivery.get_router(send_to_id);
    }
    
    ~Worker_RingReduce() {
        segment_size_arr.clear();
        segment_end_arr.clear();
    }
    
    void syncGradient(size_t epoch, T* gradPtr, std::function<void(size_t)> reduce_callback,
              std::function<void(size_t)> gather_callback) {
        // Firstly do all-reduce
        for (size_t i = 0; i < _ring_size - 1; i++) {
            const size_t send_segment_id = (cur_node_id + _ring_size - i) % _ring_size;
            const size_t recv_segment_id = (cur_node_id + _ring_size - i - 1) % _ring_size;
            
            // send segment to next-skip on the ring topology
            T* snd_ptr = gradPtr + segment_end_arr[send_segment_id] - segment_size_arr[send_segment_id];
            T* rcv_ptr = gradPtr + segment_end_arr[recv_segment_id] - segment_size_arr[recv_segment_id];
            
            step_barrier.reset(2);
            
            const size_t step_version = epoch * (2 * _ring_size - 2) + i + 1;
            // receive segment from last-skip on the ring topology
            regist_reduce_handler(rcv_ptr, recv_segment_id, step_version);
            
            PackageDescript desc(REQUEST_PUSH, step_version);
            desc.content = Buffer(snd_ptr, segment_size_arr[send_segment_id] * sizeof(T));
            
            bool pre_ahead_flag = true;
            desc.callback = [this, step_version, &pre_ahead_flag](std::shared_ptr<PackageDescript> resp_package) {
                assert(resp_package->epoch_version && resp_package->epoch_version <= step_version);
                if (resp_package->epoch_version == step_version) {
                    step_barrier.unblock(); // block until next getting into same step
                    pre_ahead_flag = false;
                }
                if (resp_package->epoch_version < step_version) {
                    assert(step_version - resp_package->epoch_version == 1);
                }
            };
            
            do {
                gDelivery.send_async(desc, send_to_id);
                if (pre_ahead_flag) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    printf("[REDUCE] wait for %zu step\n", step_version);
                }
            } while(pre_ahead_flag);
            
            step_barrier.block();
            
            if (reduce_callback) {
                reduce_callback(i);
            }
        }
        
        // Secondly do all-gather
        for (size_t i = 0; i < _ring_size - 1; i++) {
            const size_t send_segment_id = (cur_node_id + 1 + _ring_size - i) % _ring_size;
            const size_t recv_segment_id = (cur_node_id + _ring_size - i) % _ring_size;
            
            // send segment to next-skip on the ring topology
            T* snd_ptr = gradPtr + segment_end_arr[send_segment_id] - segment_size_arr[send_segment_id];
            T* rcv_ptr = gradPtr + segment_end_arr[recv_segment_id] - segment_size_arr[recv_segment_id];
            
            step_barrier.reset(2);
            
            const size_t step_version = epoch * (2 * _ring_size - 2) + _ring_size + i;
            // receive segment from last-skip on the ring topology
            regist_gather_handler(rcv_ptr, recv_segment_id, step_version);
            
            PackageDescript desc(REQUEST_PUSH, step_version);
            desc.content = Buffer(snd_ptr, segment_size_arr[send_segment_id] * sizeof(T));
            
            bool pre_ahead_flag = true;
            desc.callback = [this, step_version, &pre_ahead_flag](std::shared_ptr<PackageDescript> resp_package) {
                assert(resp_package->epoch_version && resp_package->epoch_version <= step_version);
                if (resp_package->epoch_version == step_version) {
                    step_barrier.unblock(); // block until next getting into same step
                    pre_ahead_flag = false;
                }
                if (resp_package->epoch_version < step_version) {
                    assert(step_version - resp_package->epoch_version == 1);
                }
            };
            
            do {
                gDelivery.send_async(desc, send_to_id);
                if (pre_ahead_flag) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    printf("[REDUCE] wait for %zu step\n", step_version);
                }
            } while(pre_ahead_flag);
            
            step_barrier.block();
            
            if (gather_callback) {
                gather_callback(i);
            }
        }
    }
    
private:
    void regist_reduce_handler(T* ptr, size_t recv_segment_id, size_t step_version) {
        assert(ptr);
        request_handler_t reduce_handler = [this, ptr, recv_segment_id, step_version](
                                                std::shared_ptr<PackageDescript> request,
                                                PackageDescript& response) {
            assert(request->node_id > BEGIN_ID_OF_WORKER);
            const size_t worker_id = request->node_id;
            assert(worker_id == recv_from_id);
            
            assert(request->epoch_version >= step_version);
            if (step_version != request->epoch_version) {
                response.epoch_version = step_version;
                return;
            }
            
            assert(request->content.size() % sizeof(T) == 0);
            printf("[REDUCE] step %zu: recv %zu gradients\n", step_version,
                            request->content.size() / sizeof(T));
            
            assert(segment_size_arr[recv_segment_id] * sizeof(T) == request->content.size());
            
            if (typeid(T) == typeid(float)) { // try to use AVX
                float* ptr_t = static_cast<float*>(ptr);
                const float* buffer = reinterpret_cast<const float*>(request->content.buffer());
                avx_vecAdd(buffer, ptr_t, ptr_t, segment_size_arr[recv_segment_id]);
            } else {
                T grad_value;
                size_t offset = 0;
                while (!request->content.readEOF()) {
                    request->content >> grad_value;
                    *(ptr + offset) += grad_value; // accumulate gradients
                    offset++;
                }
                assert(offset == segment_size_arr[recv_segment_id]);
            }
            response.epoch_version = step_version;
            step_barrier.unblock();
        };
        gDelivery.regist_handler(REQUEST_PUSH, std::move(reduce_handler));
    }
    
    void regist_gather_handler(T* ptr, size_t recv_segment_id, size_t step_version) {
        assert(ptr);
        request_handler_t gather_handler = [this, ptr, recv_segment_id, step_version](
                                                std::shared_ptr<PackageDescript> request,
                                                PackageDescript& response) {
            assert(request->node_id > BEGIN_ID_OF_WORKER);
            const size_t worker_id = request->node_id;
            assert(worker_id == recv_from_id);
            
            assert(request->epoch_version >= step_version);
            if (step_version != request->epoch_version) {
                response.epoch_version = step_version;
                return;
            }
            
            assert(segment_size_arr[recv_segment_id] * sizeof(T) == request->content.size());
            printf("[GATHER] step %zu: recv %zu gradients\n", step_version,
                            request->content.size() / sizeof(T));
            
            memcpy(ptr, request->content.buffer(), segment_size_arr[recv_segment_id] * sizeof(T));
            
            response.epoch_version = step_version;
            step_barrier.unblock();
        };
        gDelivery.regist_handler(REQUEST_PUSH, std::move(gather_handler));
    }
    
    std::vector<size_t> segment_size_arr;
    std::vector<size_t> segment_end_arr;
    
    size_t _param_size;
    size_t _ring_size;
    
    size_t cur_node_id;
    size_t recv_from_id;
    size_t send_to_id;
    
    Barrier step_barrier;
};

#endif /* ring_collect_h */
