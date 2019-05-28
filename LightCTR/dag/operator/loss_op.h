//
//  loss_op.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/23.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef loss_op_h
#define loss_op_h

#include "../terminus_node.h"

template <typename LossFunction>
class LossOp : public TerminusNode {
public:
    LossOp() : TerminusNode(1) {
    }
    
    float getLoss() const {
        return _loss;
    }
    
    void setLable(std::shared_ptr<std::vector<int> > label) {
        _label = label;
    }
    
protected:
    void forward_compute(const std::vector<DAG_Output>& in_outputs) {
        // compute delta via loss function
        assert(_label && in_outputs.size() == 1);
        const size_t len = in_outputs[0].data->size();
        if (node_output.data == nullptr) {
            node_output.data = std::make_shared<std::vector<float> >(len);
        }
        std::memcpy(node_output.data->data(), in_outputs[0].data->data(), len * sizeof(float));
        _loss = lossFun.loss(in_outputs[0].data->data(), _label->data(), len);
    }
    
    void backward_compute(const std::vector<DAG_Output>& out_deltas) {
        // back propagate delta
        assert(_label);
        const size_t len = node_output.data->size();
        assert(_label->size() == len);
        if (node_delta.data == nullptr) {
            node_delta.data = std::make_shared<std::vector<float> >(len);
        }
        lossFun.gradient(node_output.data->data(), _label->data(),
                         node_delta.data->data(), len);
    }
    
private:
    float _loss;
    std::shared_ptr<std::vector<int> > _label;
    LossFunction lossFun;
};

#endif /* loss_op_h */
