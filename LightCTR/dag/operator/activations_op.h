//
//  activations_op.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/23.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef activations_op_h
#define activations_op_h

#include "../../util/activations.h"
#include "string.h"

template <typename ActivationFunction>
class ActivationsOp : public AggregateNode {
public:
    ActivationsOp() = delete;
    ActivationsOp(size_t _out_cnt) : AggregateNode(1, _out_cnt) {
    }
    
protected:
    void forward_compute(const std::vector<DAG_Output>& in_outputs) {
        assert(in_outputs[0].data);
        const size_t len = in_outputs[0].data->size();
        if (node_output.data == nullptr) {
            node_output.data = std::make_shared<std::vector<float> >(len);
        }
        std::memcpy(node_output.data->data(), in_outputs[0].data->data(), len * sizeof(float));
        activFun.forward(node_output.data->data(), len);
    }
    
    void backward_compute(const std::vector<DAG_Output>& out_deltas) {
        const size_t len = out_deltas[0].data->size();
        if (node_delta.data == nullptr) {
            node_delta.data = std::make_shared<std::vector<float> >(len);
        }
        std::memcpy(node_delta.data->data(), out_deltas[0].data->data(), len * sizeof(float));
        
        activFun.backward(node_delta.data->data(), node_output.data->data(),
                          node_delta.data->data(), len);
    }
    
private:
    ActivationFunction activFun;
};

#endif /* activations_op_h */
