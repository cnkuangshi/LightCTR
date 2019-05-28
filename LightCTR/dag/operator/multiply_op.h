//
//  multiply_op.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/20.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef multiply_op_h
#define multiply_op_h

#include "string.h"
#include "../../common/avx.h"

class MultiplyOp : public AggregateNode {
public:
    MultiplyOp() = delete;
    MultiplyOp(size_t _in_cnt, size_t _out_cnt = 1) : AggregateNode(_in_cnt, _out_cnt) {
        assert(_in_cnt > 0 && _out_cnt > 0);
    }
    
protected:
    void forward_compute(const std::vector<DAG_Output>& in_outputs) {
        const size_t len = in_outputs[0].data->size();
        if (node_output.data == nullptr) {
            node_output.data = std::make_shared<std::vector<float> >(len);
        }
        std::memcpy(node_output.data->data(), in_outputs[0].data->data(), len * sizeof(float));
        for(size_t i = 1; i < in_outputs.size(); i++) {
            compute_records.push_back(in_outputs[i]);
            avx_vecScale(node_output.data->data(), node_output.data->data(),
                         len, in_outputs[i].data->data());
        }
    }
    
    void backward_compute(const std::vector<DAG_Output>& out_deltas) {
        const size_t len = out_deltas[0].data->size();
        std::vector<float> delta_arr(len, 0);
        
        for(auto& out_delta : out_deltas) {
            avx_vecAdd(out_delta.data->data(), delta_arr.data(),
                       delta_arr.data(), len);
        }
        avx_vecScale(delta_arr.data(), node_delta.data->data(),
                     len, node_output.data->data());
        
        auto& order_ids = get_in_promises_ids();
        auto& order_promises = get_in_complete_promises();
        
        for (size_t i = 0; i < order_ids.size(); i++) {
            const size_t target_id = order_ids[i];
            for (auto& record : compute_records) {
                if (record.node_id == target_id) {
                    avx_vecDiv(node_delta.data->data(), record.data->data(),
                               node_delta.data->data(), len);
                    order_promises[i].set_value(node_delta);
                    break;
                }
            }
        }
        for (auto& record : compute_records) {
            if (record.node_id == get_first_target_id()) {
                avx_vecDiv(node_delta.data->data(), record.data->data(),
                           node_delta.data->data(), len);
                return;
            }
        }
        assert(false);
    }
};

#endif /* multiply_op_h */
