//
//  node_abst.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/5.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef node_abst_h
#define node_abst_h

#include <future>
#include "../common/thread_pool.h"

class Node_Abst {
public:
    class DAG_Output {
    public:
        DAG_Output() {
            node_id = 0;
            data = nullptr;
        }
        DAG_Output(size_t _node_id) : node_id(_node_id) {
        }
        
        std::shared_ptr<std::vector<float> > data;
        
        size_t node_id;
        bool deal_flag = 0;
    };
    
    Node_Abst() = delete;
    Node_Abst(size_t _in_cnt, size_t _out_cnt) : dag_threadpool(ThreadPool::Instance()),
    in_cnt(_in_cnt), out_cnt(_out_cnt) {
        static size_t global_node_id = 1; // begin from 1
        
        node_id = global_node_id++;
        node_output.node_id = node_id; // set output node_id
        
        in_nodes.reserve(in_cnt);
    };
    
    inline void regist_in_node(std::shared_ptr<Node_Abst> ptr) {
        assert(ptr != nullptr && in_nodes.size() < in_cnt);
        in_nodes.emplace_back(ptr);
    }
    
    inline const DAG_Output& getOutput() const {
        return node_output;
    }
    
    inline size_t getNodeId() const {
        return node_id;
    }

protected:
    std::future<DAG_Output> forward_run(size_t out_id = 0) {
        assert(in_nodes.size() == in_cnt);
        
        if (node_output.deal_flag) { // return cached intermediate result
            return dag_threadpool.addTask([&]() -> DAG_Output {
                return node_output;
            });
        }
        
        if(unlikely(CAS32(&forward_flag, 0, 1))) {
            forward_reset();
            auto in_futures = std::make_shared<std::vector<std::future<DAG_Output> > >(in_cnt);
            if (in_cnt > 0) { // source_node in_cnt == 0
                for(size_t i = 0; i < in_cnt; i++) {
                    in_futures->at(i) = in_nodes[i]->forward_run(getNodeId());
                }
            }
            return dag_threadpool.addTask([&, in_futures]() -> DAG_Output {
                std::vector<DAG_Output> in_outputs(in_cnt);
                if (in_cnt > 0) {
                    for(size_t i = 0; i < in_cnt; i++) {
                        in_outputs[i] = in_futures->at(i).get();
                        assert(in_outputs[i].node_id > 0);
                    }
                }
                return forward_compute_wrapper(in_outputs);
            });
        };
        assert(out_id_inc < out_cnt - 1);
        return out_complete_promises[out_id_inc++].get_future();
    }
    
    virtual void forward_compute(const std::vector<DAG_Output>&) = 0;
    
    void init_forward_Flow(bool keep_intermediate) {
        assert(in_nodes.size() == in_cnt);
        forward_flag = 0; // force reset first targeting
        
        if (!keep_intermediate) {
            node_output.deal_flag = false;
        }
        for(auto& in_node : in_nodes) {
            in_node->init_forward_Flow(keep_intermediate);
        }
    }
    
    inline void complete_out_promises() {
        for (auto& promise : out_complete_promises) {
            promise.set_value(node_output);
        }
    }
    
    size_t in_cnt, out_cnt;
    uint32_t forward_flag = 0;
    
    DAG_Output node_output;
    std::vector<DAG_Output> compute_records;
    
    ThreadPool& dag_threadpool;
    
private:
    DAG_Output forward_compute_wrapper(const std::vector<DAG_Output>& out_deltas) {
        forward_compute(out_deltas);
        node_output.deal_flag = true;
        
        complete_out_promises();
        return node_output;
    }
    
    void forward_reset() {
        out_id_inc = 0;
        compute_records.clear(); // clear records when forward reset
        if (likely(out_cnt > 0)) { // terminus_node out_cnt == 0
            out_complete_promises.clear();
            out_complete_promises.reserve(out_cnt - 1);
            for (size_t i = 0; i < out_cnt - 1; i++) {
                out_complete_promises.emplace_back(std::promise<DAG_Output>());
            }
        }
    }
    
    std::vector<std::promise<DAG_Output> > out_complete_promises;
    
    std::vector<std::shared_ptr<Node_Abst> > in_nodes;
    
    size_t out_id_inc = 0;
    size_t node_id;
};


class Autograd_Node_Abst : public Node_Abst {
public:
    Autograd_Node_Abst() = delete;
    Autograd_Node_Abst(size_t _in_cnt, size_t _out_cnt) : Node_Abst(_in_cnt, _out_cnt) {
        node_delta.node_id = getNodeId(); // set delta node_id
        
        out_nodes.reserve(out_cnt);
    };
    
    inline void regist_out_node(std::shared_ptr<Autograd_Node_Abst> ptr) {
        assert(ptr != nullptr && out_nodes.size() < out_cnt);
        out_nodes.emplace_back(ptr);
    }
    
    inline const DAG_Output& getDelta() const {
        return node_delta;
    }
    
protected:
    std::future<DAG_Output> backward_run(size_t in_id = 0) {
        assert(out_nodes.size() == out_cnt);
        
        if (node_delta.deal_flag) { // return cached intermediate result
            return dag_threadpool.addTask([&]() -> DAG_Output {
                return node_delta;
            });
        }
        
        if(unlikely(CAS32(&backward_flag, 0, 1))) {
            backward_reset();
            first_target_id = in_id;
            auto out_futures = std::make_shared<std::vector<std::future<DAG_Output> > >(out_cnt);
            if (out_cnt > 0) {
                for(size_t i = 0; i < out_cnt; i++) {
                    out_futures->at(i) = out_nodes[i]->backward_run(getNodeId());
                }
            }
            return dag_threadpool.addTask([&, out_futures]() -> DAG_Output {
                std::vector<DAG_Output> out_deltas(out_cnt);
                if (out_cnt > 0) { // terminus_node out_cnt == 0
                    for(size_t i = 0; i < out_cnt; i++) {
                        out_deltas[i] = out_futures->at(i).get();
                        assert(out_deltas[i].node_id > 0);
                    }
                }
                return backward_compute_wrapper(out_deltas);
            });
        };
        assert(in_id_inc < in_cnt - 1);
        in_promises_ids.emplace_back(in_id);
        return in_complete_promises[in_id_inc++].get_future();
    }
    
    virtual void backward_compute(const std::vector<DAG_Output>&) = 0;
    
    void init_backward_Flow(bool keep_intermediate) {
        assert(out_nodes.size() == out_cnt);
        backward_flag = 0; // force reset first targeting
        
        if (!keep_intermediate) {
            node_delta.deal_flag = false;
        }
        for(auto& out_node : out_nodes) {
            out_node->init_backward_Flow(keep_intermediate);
        }
    }
    
    inline void complete_in_promises() {
        for (auto& promise : in_complete_promises) {
            promise.set_value(node_delta);
        }
    }
    
    inline const std::vector<size_t>& get_in_promises_ids() const {
        return in_promises_ids;
    }
    
    inline std::vector<std::promise<DAG_Output> >& get_in_complete_promises() {
        return in_complete_promises;
    }
    
    inline size_t get_first_target_id() const {
        assert(first_target_id > 0);
        return first_target_id;
    }
    
    DAG_Output node_delta;
    
private:
    DAG_Output backward_compute_wrapper(const std::vector<DAG_Output>& out_deltas) {
        backward_compute(out_deltas);
        node_delta.deal_flag = true;
        
        complete_in_promises();
        return node_delta;
    }
    
    void backward_reset() {
        in_id_inc = 0;
        if (likely(in_cnt > 0)) { // source_node in_cnt == 0
            in_promises_ids.clear();
            in_promises_ids.reserve(in_cnt - 1);
            
            in_complete_promises.clear();
            in_complete_promises.reserve(in_cnt - 1);
            for (size_t i = 0; i < in_cnt - 1; i++) {
                in_complete_promises.emplace_back(std::promise<DAG_Output>());
            }
        }
    }
    
    std::vector<std::promise<DAG_Output> > in_complete_promises;
    std::vector<size_t> in_promises_ids;
    size_t first_target_id; // record the first targeting node_id
    
    std::vector<std::shared_ptr<Autograd_Node_Abst> > out_nodes;
    
    uint32_t backward_flag = 0;
    size_t in_id_inc = 0;
};

#endif /* node_abst_h */
