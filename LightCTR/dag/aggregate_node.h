//
//  aggregate_node.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/19.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef aggregate_node_h
#define aggregate_node_h

#include <vector>
#include "node_abst.h"

// Aggregate or Scatter Flow
class AggregateNode : public Autograd_Node_Abst {
public:
    AggregateNode() = delete;
    AggregateNode(size_t _in_cnt, size_t _out_cnt = 1) : Autograd_Node_Abst(_in_cnt, _out_cnt) {
        assert(_in_cnt > 0 && _out_cnt > 0);
    }
    
protected:
    virtual void forward_compute(const std::vector<DAG_Output>& in_outputs) = 0;
    
    virtual void backward_compute(const std::vector<DAG_Output>& out_deltas) = 0;
};

#endif /* aggregate_node_h */
