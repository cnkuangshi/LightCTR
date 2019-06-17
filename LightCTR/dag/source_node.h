//
//  source_node.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/19.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef source_node_h
#define source_node_h

#include <vector>
#include "string.h"
#include "node_abst.h"
#include "../common/avx.h"

class SourceNode : public Autograd_Node_Abst {
public:
    SourceNode() = delete;
    explicit SourceNode(size_t _out_cnt) : Autograd_Node_Abst(0, _out_cnt) {
        assert(_out_cnt > 0);
    }
    
    DAG_Output runFlow(bool keep_intermediate = false) {
        init_backward_Flow(keep_intermediate);
        return backward_run().get();
    }
    
    void setValue(std::shared_ptr<std::vector<float> > data) {
        node_output.data = data;
    }

protected:
    virtual void forward_compute(const std::vector<DAG_Output>&) {
        // provide value of data source
    }
    
    virtual void backward_compute(const std::vector<DAG_Output>& out_deltas) {
        // apply delta as gradient on the value
    }
};


template <typename UpdaterFunc>
class TrainableNode : public SourceNode {
public:
    TrainableNode() = delete;
    explicit TrainableNode(size_t _out_cnt) : SourceNode(_out_cnt) {
        assert(_out_cnt > 0);
    }
    
    void setValue(std::shared_ptr<std::vector<float> > data) {
        node_output.data = data;
        updater.learnable_params_cnt(data->size());
    }
    
protected:
    void forward_compute(const std::vector<DAG_Output>&) {
        // provide value of data source
        assert(node_output.data);
    }
    
    void backward_compute(const std::vector<DAG_Output>& out_deltas) {
        // apply delta as gradient on the value
        const size_t len = out_deltas[0].data->size();
        assert(len == node_output.data->size());
        
        if (node_delta.data == nullptr) {
            node_delta.data = std::make_shared<std::vector<float> >(len);
        }
        std::memset(node_delta.data->data(), 0, len * sizeof(float));
        for(auto& out_delta : out_deltas) {
            avx_vecAdd(node_delta.data->data(), out_delta.data->data(),
                       node_delta.data->data(), len);
        }
        updater.update(0, len, node_output.data->data(), node_delta.data->data());
    }
private:
    UpdaterFunc updater;
};

#endif /* source_node_h */
