//
//  train_gbm_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/26.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_gbm_algo.h"
#include "unistd.h"
#include <algorithm>

void Train_GBM_Algo::init() { // run once per gbm train stage
    eps_feature_value = 1e-8;
    lambda = 1e-5;
    learning_rate = 0.6f;
    
    has_pred_tree = -1;
    dataSet_Pred = new double[this->dataRow_cnt];
    memset(dataSet_Pred, 0, sizeof(double) * this->dataRow_cnt);
    
    sampleDataSetIndex = new bool[this->dataRow_cnt];
    sampleFeatureSetIndex = new bool[this->feature_cnt];
    dataRow_LocAtTree = new RegTreeNode*[this->dataRow_cnt];
    dataSet_Grad.clear();
    
    splitNodeStat_thread = new SplitNodeStat_Thread[((1<<this->maxDepth) - 1) * this->proc_cnt];
}

void Train_GBM_Algo::flash(RegTreeNode *root) { // run per gbm tree building
    assert(root != NULL && this->leafNodes_tmp.size() == 1);
    
    sample(); // sample dataRow and feature for each tree
    
    for (size_t rid = 0; rid < this->dataRow_cnt; rid++) {
        if (!sampleDataSetIndex[rid]) {
            continue;
        }
        // put all data belong to tree root
        dataRow_LocAtTree[rid] = root;
        root->leafStat->data_cnt++;
        
        // predict based on prev RegTree and calculate new predict without current root
        for (size_t tid = max(has_pred_tree, 0); tid < RegTreeRootArr.size() - 1; tid++) {
            double w = locAtLeafWeight(RegTreeRootArr[tid], dataSet[rid]);
            dataSet_Pred[rid] += learning_rate * w;
        }
        assert(dataSet_Pred[rid] < 30 || dataSet_Pred[rid] > -30);
        double pred = activFunc(dataSet_Pred[rid]);
        
        // calculate node's total sum of all data's grad and hess
        pair<double, double> grad_pair = make_pair(grad(pred, label[rid]), hess(pred));
        assert(grad_pair.second >= 0);
        dataSet_Grad[rid] = grad_pair;
        assert(root->leafStat == this->leafNodes_tmp.front());
        this->leafNodes_tmp.front()->sumGrad += grad_pair.first;
        this->leafNodes_tmp.front()->sumHess += grad_pair.second;
    }
    has_pred_tree++;
}

void Train_GBM_Algo::Train() {
    for (size_t i = 0; i < this->epoch_cnt; i++) {
        cout <<"Training Tree = " << i << endl;
        
        RegTreeNode *root = newTree();
        
        // train new Tree
        flash(root);
        for (size_t depth = 1; depth <= this->maxDepth; depth++) {
            
            // swap leafNodes to new Array
            swap(this->leafNodes, this->leafNodes_tmp);
            this->leafNodes_tmp.clear();
            
            size_t feature_thread_hold = (this->dataSet_feature.size() + this->proc_cnt - 1) / this->proc_cnt;
            
            // multithread to find different feature's split point
            this->proc_left = (int)this->feature_cnt * 2;
            
            threadpool->init();
            for (size_t j = 0; j < this->proc_cnt; j++) {
                size_t start_pos = j * feature_thread_hold;
                threadpool->addTask(bind(&Train_GBM_Algo::findSplitFeature_Wrapper, this, start_pos, min(start_pos + feature_thread_hold, this->dataSet_feature.size()), j));
            }
            threadpool->join();
            assert(proc_left == 0);
            
            // global to gather leafNodes' best split point of all threads
            for (auto it = leafNodes.begin(); it != leafNodes.end(); it++) {
                if ((*it)->data_cnt == 0) { // filter non-data LeafNodes
                    continue;
                }
                size_t node_id = (*it)->treeNode->node_index;
                assert(node_id >= 0);
                for (size_t pid = 0; pid < this->proc_cnt; pid++) {
                    SplitNodeStat_Thread *stat = &splitNodeStat_thread[pid * ((1<<this->maxDepth) - 1) + node_id];
                    if ((*it)->needUpdate(stat->gain, stat->split_feature_index)) {
                        // best split feature&threshold update to leafNodes->RegTreeNode
                        (*it)->treeNode->split_feature_index = stat->split_feature_index;
                        (*it)->treeNode->split_threshold = stat->split_threshold;
                        (*it)->gain = stat->gain;
                        (*it)->treeNode->dataNAN_go_Right = stat->dataNAN_go_Right;
                    }
                    // clear splitNodeStat_thread's gain & split info, init state for next iter
                    stat->gain = 0, stat->split_feature_index = -1, stat->split_threshold = 0;
                    stat->clear();
                }
                // update dataRow_LocAtTree
                if ((*it)->gain == 0 || depth == this->maxDepth) {
                    // weight less than minLeafW or get max depth, ture tree node into un-active leaf
                    turn_leaf((*it)->treeNode);
                    double w = weight((*it)->sumGrad, (*it)->sumHess);
                    (*it)->treeNode->leafStat->weight = w;
                } else {
                    // split tree node and divide node's sample data into new node
                    split_node((*it)->treeNode);
                    for (size_t rid = 0; rid < this->dataRow_cnt; rid++) {
                        if (!sampleDataSetIndex[rid]) {
                            continue;
                        }
                        assert(dataRow_LocAtTree[rid] != NULL);
                        if (bLeaf(dataRow_LocAtTree[rid])) { // skip data has been in leaf
                            continue;
                        }
                        dataRow_LocAtTree[rid] = nextLevel(dataRow_LocAtTree[rid], dataSet[rid]);
                        assert(dataRow_LocAtTree[rid]->leafStat != NULL);
                        dataRow_LocAtTree[rid]->leafStat->data_cnt++;
                    }
                }
            }
//            for (auto it = leafNodes.begin(); it != leafNodes.end(); it++) {
//                auto node = (*it)->treeNode;
//                printf("--- Node %zu have %zu rows using threshold %lf %d active=%d\n", node->node_index, (*it)->data_cnt, node->split_threshold, node->split_feature_index, node->leafStat == NULL ? 1 : 0);
//            }
            if (this->leafNodes_tmp.empty()) {
                // none point to split, break
                break;
            }
            
            // update next level leafNodes_tmp's dataRows' sumGrad & sumHess
            for (size_t rid = 0; rid < this->dataRow_cnt; rid++) {
                if (!sampleDataSetIndex[rid]) {
                    continue;
                }
                RegTreeNode* node = dataRow_LocAtTree[rid];
                
                if (!node->leafStat->active) {
                    // skip data has been in leaf, only new split LeafNodes are active
                    continue;
                }
                pair<double, double> pair = dataSet_Grad[rid];
                node->leafStat->sumGrad += pair.first;
                node->leafStat->sumHess += pair.second;
            }
        }
        // TODO backward pruning tree
    }
}

void Train_GBM_Algo::findSplitFeature_Wrapper(size_t rbegin, size_t rend, size_t pid) {
    // from min left to max right, default put NAN data into right
    findSplitFeature(rbegin, rend, pid, 0);
    // from max right to min left, default put NAN data into left
    findSplitFeature(rbegin, rend, pid, 1);
}

void Train_GBM_Algo::findSplitFeature(size_t rbegin, size_t rend, size_t pid, bool dataNAN_go_Right) {
    for (size_t fid = rbegin; fid < rend; fid++) {
        if (!sampleFeatureSetIndex[fid]) {
            continue;
        }
        assert(this->dataSet_feature[fid].size() > 0);
        sort(this->dataSet_feature[fid].begin(), this->dataSet_feature[fid].end());
        
        auto dataRow_in_fea = dataSet_feature[fid];
        if (!dataNAN_go_Right) {
            reverse(dataRow_in_fea.begin(), dataRow_in_fea.end());
        }
        auto begin = dataRow_in_fea.begin();
        auto end = dataRow_in_fea.end();
        
        // calc all data which contain the feature, whether data can be best split point in its three node
        for (auto it = begin; it != end; it++) {
            size_t rid = it->first;
            if (!sampleDataSetIndex[rid]) {
                continue;
            }
            RegTreeNode* node = dataRow_LocAtTree[rid];
            
            if (!node->leafStat->active) { // skip leaf data
                continue;
            }
            
            size_t node_id = node->node_index;
            assert(node_id >= 0);
            SplitNodeStat_Thread *stat = &splitNodeStat_thread[pid * ((1<<this->maxDepth) - 1) + node_id];
            double value = it->second;
            
            if (stat->sumHess == 0) {
                // first data for one node, pass
                assert(((stat->split_feature_index == -1) ^ (stat->split_feature_index != -1 && stat->gain != 0)) == 1);
                assert(stat->sumHess == 0 && stat->sumGrad == 0 && stat->last_value_toCheck == 1e-12);
            } else {
                if (fabs(value - stat->last_value_toCheck) > eps_feature_value) {
                    assert(stat->sumHess >= 0);
                    if (stat->sumHess > minLeafW) {
                        LeafNodeStat* globalLeafStat = node->leafStat;
                        assert(globalLeafStat != NULL);
                        double leftPartGain = gain(stat->sumGrad, stat->sumHess);
                        double rightPartGain = gain(globalLeafStat->sumGrad - stat->sumGrad, globalLeafStat->sumHess - stat->sumHess);
                        double splitGain = leftPartGain + rightPartGain - gain(globalLeafStat->sumGrad, globalLeafStat->sumHess);
                        
                        if (stat->needUpdate(splitGain, fid)) {
                            stat->split_feature_index = (int)fid;
                            stat->split_threshold = value + (dataNAN_go_Right ? 1e-11f : -1e-11f);
                            stat->gain = splitGain; // for global select max Gain
                            stat->dataNAN_go_Right = dataNAN_go_Right;
                        }
                    }
                }
            }
            pair<double, double> pair = dataSet_Grad[rid];
            stat->sumGrad += pair.first;
            stat->sumHess += pair.second;
            assert(stat->sumHess > 0);
            assert(node->leafStat->sumHess + 1e-10 >= stat->sumHess);
            stat->last_value_toCheck = value;
        }
        // calculate gain when all NAN dataRow goto one direction and all data contain the feature goto another direction, at the same time some LeafNodes' splitGain equal 0 because all data don't contain the feature
        for (auto it = leafNodes.begin(); it != leafNodes.end(); it++) {
            size_t node_id = (*it)->treeNode->node_index;
            assert(node_id >= 0);
            SplitNodeStat_Thread *stat = &splitNodeStat_thread[pid * ((1<<this->maxDepth) - 1) + node_id];
            if (stat->gain == 0) { // the leafNode don't split by this feature
                stat->clear();
                continue;
            }
            double leftPartGain = gain(stat->sumGrad, stat->sumHess);
            double rightPartGain = gain((*it)->sumGrad - stat->sumGrad, (*it)->sumHess - stat->sumHess);
            double splitGain = leftPartGain + rightPartGain - gain((*it)->sumGrad, (*it)->sumHess);
            
            if (stat->needUpdate(splitGain, fid)) {
                assert(splitGain > 0);
                stat->split_feature_index = (int)fid;
                stat->split_threshold = stat->last_value_toCheck;
                dataNAN_go_Right ? stat->split_threshold += 1e-11f : stat->split_threshold -= 1e-11f;
                stat->gain = splitGain;
                stat->dataNAN_go_Right = dataNAN_go_Right;
            }
            // clear sumGrad and sumHess for next feature accumulate
            stat->clear();
        }
    }
    {
        unique_lock<mutex> glock(this->lock);
        assert(this->proc_left > 0);
        proc_left -= (rend - rbegin);
    }
}

