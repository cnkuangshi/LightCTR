//
//  matmul_op.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/24.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef matmul_op_h
#define matmul_op_h

#include "../../common/avx.h"

class MatmulOp : public AggregateNode {
public:
    MatmulOp() = delete;
    MatmulOp(size_t _out_cnt) : AggregateNode(2, _out_cnt) {
        assert(_out_cnt > 0);
    }
    
protected:
    void forward_compute(const std::vector<DAG_Output>& in_outputs) {
        assert(in_outputs.size() == 2);
        if (node_output.data == nullptr) {
            node_output.data = std::make_shared<std::vector<float> >(1);
        }
        compute_records.push_back(in_outputs[0]);
        compute_records.push_back(in_outputs[1]);
        node_output.data->at(0) = avx_dotProduct(in_outputs[0].data->data(),
                                                 in_outputs[1].data->data(),
                                                 in_outputs[0].data->size());
    }
    
    void backward_compute(const std::vector<DAG_Output>& out_deltas) {
        float cur_delta = 0;
        for(auto& out_delta : out_deltas) {
            cur_delta += out_delta.data->at(0);
        }
        
        assert(compute_records.size() == 2);
        const size_t len = compute_records[0].data->size();
        if (node_delta.data == nullptr) {
            node_delta.data = std::make_shared<std::vector<float> >(len);
        }
        
        auto& order_promises = get_in_complete_promises();
        assert(order_promises.size() == 1);
        
        size_t index = 0;
        if (compute_records[1].node_id == get_first_target_id()) {
            index = 1;
        }
        
        avx_vecScale(compute_records[index].data->data(),
                     node_delta.data->data(),
                     len, cur_delta);
        order_promises[0].set_value(node_delta);
        
        // Notice to remove targeted promise for repeating set promise value
        // otherwise, it will be "terminating with uncaught exception of type
        // std::__1::future_error: The state of the promise has already been set"
        order_promises.clear();
        
        avx_vecScale(compute_records[1 - index].data->data(),
                     node_delta.data->data(),
                     len, cur_delta);
    }
};

#endif /* matmul_op_h */
