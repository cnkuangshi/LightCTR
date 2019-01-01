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

const size_t kPreaHeadWaitInterval = 50;

// especially design for GPUs' collective ring-reduce
template<typename T>
class Worker_RingReduce : public Dist_Machine_Abst {
public:
    Worker_RingReduce(BufferFusion<T>& buf_fusion, size_t ring_size)
                    : _buf_fusion(buf_fusion), _ring_size(ring_size) {
        _param_size = _buf_fusion.size();
        assert(_param_size > 0 && ring_size > 0);
        assert(ring_size == __global_cluster_worker_cnt);
                        
        segment_size_arr.resize(ring_size);
        segment_end_arr.resize(ring_size);
        const size_t seg_size = _param_size / ring_size;
        const size_t seg_res = _param_size % ring_size;
        
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
        // check router
        gDelivery.get_router(recv_from_id);
        gDelivery.get_router(send_to_id);
                        
        // TODO recovering boot mode, Copy parameters from the other conventional Ring worker
        // and Start send and receive by processing epoch_version
    }
    
    ~Worker_RingReduce() {
        segment_size_arr.clear();
        segment_end_arr.clear();
    }
    
    void syncGradient(size_t epoch,
                      std::function<void(size_t)> reduce_callback = NULL,
                      std::function<void(size_t)> gather_callback = NULL) {
        // Firstly do all-reduce
        for (size_t i = 0; i < _ring_size - 1; i++) {
            const size_t send_segment_id = (cur_node_id + _ring_size - i) % _ring_size;
            const size_t recv_segment_id = (cur_node_id + _ring_size - i - 1) % _ring_size;
            
            // send segment to next-skip on the ring topology
            size_t rcv_offset = segment_end_arr[recv_segment_id] - segment_size_arr[recv_segment_id];
            
            step_barrier.reset(2);
            
            const size_t step_version = epoch * (2 * _ring_size - 2) + i + 1;
            // receive segment from last-skip on the ring topology
            regist_reduce_handler(rcv_offset, recv_segment_id, step_version);
            
            PackageDescript desc(REQUEST_PUSH, step_version);
            Buffer* buffer_ptr = nullptr;
            _buf_fusion.memcpy_out(&buffer_ptr,
                              segment_end_arr[send_segment_id] - segment_size_arr[send_segment_id],
                              segment_size_arr[send_segment_id]);
            desc.content = std::move(*buffer_ptr);
            
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
                gDelivery.send_sync(desc, send_to_id);
                if (pre_ahead_flag) {
                    // TODO dynamic waiting interval for network delay or crash of some machines
                    std::this_thread::sleep_for(std::chrono::milliseconds(kPreaHeadWaitInterval));
                    printf("[REDUCE] waiting for the %zu step\n", step_version);
                }
            } while(pre_ahead_flag);
            
            step_barrier.block();
            
            if (unlikely(reduce_callback)) {
                reduce_callback(i);
            }
        }
        
        // Secondly do all-gather
        for (size_t i = 0; i < _ring_size - 1; i++) {
            const size_t send_segment_id = (cur_node_id + 1 + _ring_size - i) % _ring_size;
            const size_t recv_segment_id = (cur_node_id + _ring_size - i) % _ring_size;
            
            // send segment to next-skip on the ring topology
            size_t rcv_offset = segment_end_arr[recv_segment_id] - segment_size_arr[recv_segment_id];
            
            step_barrier.reset(2);
            
            const size_t step_version = epoch * (2 * _ring_size - 2) + _ring_size + i;
            // receive segment from last-skip on the ring topology
            regist_gather_handler(rcv_offset, recv_segment_id, step_version);
            
            PackageDescript desc(REQUEST_PUSH, step_version);
            Buffer* buffer_ptr = nullptr;
            _buf_fusion.memcpy_out(&buffer_ptr,
                                    segment_end_arr[send_segment_id] - segment_size_arr[send_segment_id],
                                    segment_size_arr[send_segment_id]);
            desc.content = std::move(*buffer_ptr);
            
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
                gDelivery.send_sync(desc, send_to_id);
                if (pre_ahead_flag) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(kPreaHeadWaitInterval));
                    printf("[GATHER] waiting for the %zu step\n", step_version);
                }
            } while(pre_ahead_flag);
            
            step_barrier.block();
            
            if (unlikely(gather_callback)) {
                gather_callback(i);
            }
        }
    }
    
private:
    void regist_reduce_handler(size_t rcv_offset, size_t recv_segment_id, size_t step_version) {
        request_handler_t reduce_handler = [this, rcv_offset, recv_segment_id, step_version](
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
            
            // accumulate gradients
            if (typeid(T) == typeid(float)) { // try to use AVX
                const float* buffer = reinterpret_cast<const float*>(request->content.buffer());
                
                _buf_fusion.transform(rcv_offset,
                                       segment_size_arr[recv_segment_id],
                                       [&buffer](T* begin, T* end) {
                                           avx_vecAdd(buffer, begin, begin, end - begin);
                                           buffer += end - begin;
                                       });
            } else {
                
                _buf_fusion.transform(rcv_offset,
                                       segment_size_arr[recv_segment_id],
                                       [request](T* begin, T* end) {
                                           T grad_value;
                                           for (size_t i = 0; i < end - begin; i++) {
                                               request->content >> grad_value;
                                               *(begin + i) += grad_value;
                                           }
                                       });
                assert(request->content.readEOF());
            }
            response.epoch_version = step_version;
            step_barrier.unblock();
        };
        gDelivery.regist_handler(REQUEST_PUSH, std::move(reduce_handler));
    }
    
    void regist_gather_handler(size_t rcv_offset, size_t recv_segment_id, size_t step_version) {
        request_handler_t gather_handler = [this, rcv_offset, recv_segment_id, step_version](
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
            
            const float* buffer = reinterpret_cast<const float*>(request->content.buffer());
            _buf_fusion.memcpy_in(rcv_offset, buffer, segment_size_arr[recv_segment_id]);
            
            response.epoch_version = step_version;
            step_barrier.unblock();
        };
        gDelivery.regist_handler(REQUEST_PUSH, std::move(gather_handler));
    }
    
    std::vector<size_t> segment_size_arr;
    std::vector<size_t> segment_end_arr;
    
    BufferFusion<T>& _buf_fusion;
    size_t _param_size;
    size_t _ring_size;
    
    size_t cur_node_id;
    size_t recv_from_id;
    size_t send_to_id;
    
    Barrier step_barrier;
};

#endif /* ring_collect_h */
