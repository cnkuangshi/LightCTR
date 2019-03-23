//
//  ann_index.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/5/4.
//  Copyright © 2018年 SongKuangshi. All rights reserved.
//

#ifndef ann_index_h
#define ann_index_h

#include <vector>
#include "../util/random.h"
#include "assert.h"
#include <queue>
#include "../common/avx.h"

class ANNIndex {
    typedef vector<float> Point;
    class Node;
    typedef vector<Node> Tree;
    
    class Node {
    public:
        size_t node_id;
        size_t father, left, right;
        Point hyperplane;
        float bias;
        vector<size_t>* innerPoints; // only reserve for leaf
        
        Node(const Node&& _node) {
            node_id = _node.node_id;
            father = _node.father;
            left = _node.left, right = _node.right;
            bias = _node.bias;
            hyperplane = std::move(_node.hyperplane);
            innerPoints = NULL;
            
            if (_node.innerPoints != NULL) {
                innerPoints = new vector<size_t>();
                *innerPoints = std::move(*_node.innerPoints);
            }
        }
        Node &operator=(const Node &) = delete;
        Node(Tree* tree, Node* fatherNode = NULL, bool bLeftLeaf = true) {
            left = right = 0;
            innerPoints = NULL;
            hyperplane.clear();
            bias = 0;
            node_id = tree->size();
            if (fatherNode == NULL) {
                father = 0; // be root of tree
            } else {
                father = fatherNode->node_id;
                bLeftLeaf ? (fatherNode->left = node_id) : (fatherNode->right = node_id);
            }
            assert(father < (size_t)(-1) / 8);
            tree->emplace_back(std::move(*this));
        }
        
        inline bool bRoot() const {
            return node_id == 0 && innerPoints == NULL && hyperplane.size() > 0;
        }
        inline bool bLeaf() const {
            return left == 0 && right == 0 && innerPoints != NULL;
        }
        
        inline void saveHyperplane(Point& _hyperplane, float _bias) {
            bias = _bias;
            hyperplane = std::move(_hyperplane);
        }
        inline void saveInnerPoints(const vector<size_t>& indices) {
            assert(indices.size() > 0);
            innerPoints = new vector<size_t>();
            innerPoints->resize(indices.size());
            innerPoints->assign(indices.begin(), indices.end());
        }
    };
    
public:
    ANNIndex(string dataPath, size_t _feature_cnt, size_t tree_cnt = 20) :
        feature_cnt(_feature_cnt) {
        assert(tree_cnt > 0);
        loadDataRow(dataPath, feature_cnt);
        
        vector<size_t> indices;
        indices.resize(points_set.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = i;
        }
        forest.resize(tree_cnt);
        for (size_t i = 0; i < tree_cnt; i++) {
            forest[i] = new Tree();
            forest[i]->reserve(points_set.size());
            Node root(forest[i]);
            buildTree(forest[i], &forest[i]->at(0), indices);
            assert(forest[i]->at(0).bRoot());
            assert(forest[i]->back().bLeaf());
            printf("%zu-th Tree has %lu nodes\n", i, forest[i]->size());
        }
    }
    
    ~ANNIndex() {
        for (auto it = forest.begin(); it != forest.end(); it++) {
            (*it)->clear();
        }
        forest.clear();
    }
    
    void query(const Point& input, size_t beamSearchK, vector<size_t>& result_indices) {
        assert(beamSearchK >= leaf_min_points);
        
        result_indices.reserve(beamSearchK);
        for (size_t i = 0; i < forest.size(); i++) {
            searchTree(input, forest[i], beamSearchK, result_indices);
            printf("Tree %lu - %lu\n", i, result_indices.size());
        }
        for (int i = 0; i < min(beamSearchK, result_indices.size()); i++) {
            assert(result_indices[i] < 99999);
            printf("%d ", (int)result_indices[i]);
        }
        puts("");
    }
    
private:
    void loadDataRow(string dataPath, size_t feature_cnt) {
        points_set.clear();
        
        ifstream fin_;
        string line;
        int nchar;
        float val;
        fin_.open(dataPath, ios::in);
        if(!fin_.is_open()){
            cout << "open file error!" << endl;
            exit(1);
        }
        Point tmp;
        tmp.reserve(feature_cnt);
        while(!fin_.eof()){
            getline(fin_, line);
            tmp.clear();
            const char *pline = line.c_str();
            while(pline < line.c_str() + (int)line.length() &&
                  sscanf(pline, "%f%n", &val, &nchar) >= 1){
                pline += nchar + 1;
                assert(!isnan(val));
                tmp.emplace_back(val);
                if (tmp.size() == feature_cnt) {
                    assert(tmp.size() == feature_cnt);
                    points_set.emplace_back(tmp);
                    tmp.clear();
                }
            }
        }
        assert(points_set.size() > 0);
    }
    
    inline void buildTree(Tree* tree, Node* curNode, const vector<size_t>& indices) {
        assert(!indices.empty());
        size_t curNodeId = curNode->node_id;
        
        if (indices.size() <= leaf_min_points) {
            curNode->saveInnerPoints(indices);
            assert(curNode->bLeaf());
            return;
        }
        Point hyperplane;
        hyperplane.resize(feature_cnt);
        float bias = split_twoPart(indices, hyperplane);
        
        vector<size_t> leftSet, rightSet;
        leftSet.reserve(feature_cnt);
        rightSet.reserve(feature_cnt);
        for (size_t i = 0; i < indices.size(); i++) {
            size_t index = indices[i];
            if (side(hyperplane, points_set[index], bias)) {
                leftSet.emplace_back(index);
            } else {
                rightSet.emplace_back(index);
            }
        }
        curNode->saveHyperplane(hyperplane, bias);
        assert(leftSet.size() > 0 && rightSet.size() > 0);
        
        Node leftNode(tree, curNode, 1);
        buildTree(tree, &tree->back(), leftSet);
        curNode = &tree->at(curNodeId); // fix pointer to avoid realloc memory
        
        Node rightNode(tree, curNode, 0);
        buildTree(tree, &tree->back(), rightSet);
        
        curNode = &tree->at(curNodeId);
        assert(curNode->hyperplane.size() > 0);
        return;
    }
    
    inline void searchTree(const Point& input, const Tree* tree,
                           size_t beamSearchK, vector<size_t>& result) {
        assert(!tree->empty() && tree->at(0).bRoot());
        priority_queue<pair<float, size_t> > Q;
        Q.push(make_pair(0x7fffffff, 0));
        while (result.size() < beamSearchK && !Q.empty()) {
            const auto top = Q.top();
            Q.pop();
            assert(top.first > 0);
            auto curNode = &tree->at(top.second);
            if (curNode->bLeaf()) {
                const auto pointSet = curNode->innerPoints;
                result.insert(result.end(), pointSet->begin(), pointSet->end());
            } else {
                assert(curNode->innerPoints == NULL && !curNode->hyperplane.empty());
                
                float dis = margin(curNode->hyperplane, input, curNode->bias);
                
                if (dis > 0) {
                    Q.push(make_pair(min(top.first, dis), curNode->left));
                } else {
                    Q.push(make_pair(min(top.first, -dis), curNode->right));
                }
            }
        }
    }
    
    inline float split_twoPart(const vector<size_t>& indices, Point& hyperplane) {
        int iteration_steps = 200;
        
        size_t count = indices.size();
        size_t i = Random_index(count);
        size_t j = Random_index(count - 1);
        j += (j >= i); // ensure that i != j
        Point centroid1(points_set[indices[i]]); // copy vector
        Point centroid2(points_set[indices[j]]);
        
        int ic = 1, jc = 1;
        while(iteration_steps--) {
            size_t k = Random_index(count); // random select one point to adjust centroids
            size_t index = indices[k];
            float dis1 = ic * Euclidean_distance(centroid1, points_set[index]);
            float dis2 = jc * Euclidean_distance(centroid2, points_set[index]);
            if (dis1 < dis2) {
                avx_vecScalerAdd(points_set[index].data(), centroid1.data(),
                                 centroid1.data(), ic, centroid1.size());
                avx_vecScale(centroid1.data(), centroid1.data(),
                             centroid1.size(), 1.0 / (ic + 1));
                ic++;
            } else if (dis1 > dis2) {
                avx_vecScalerAdd(points_set[index].data(), centroid2.data(),
                                 centroid2.data(), jc, centroid2.size());
                avx_vecScale(centroid2.data(), centroid2.data(),
                             centroid2.size(), 1.0 / (jc + 1));
                jc++;
            } else {
                // do nothing
            }
        }
        
        // compute normal direction of hyperplane and normalize to avoid repeat calculate
        avx_vecScalerAdd(centroid1.data(), centroid2.data(),
                         hyperplane.data(), -1, hyperplane.size());
        float norm = avx_L2Norm(hyperplane.data(), hyperplane.size());
        norm = 1.0 / sqrt(norm);
        avx_vecScale(hyperplane.data(), hyperplane.data(), hyperplane.size(), norm);
        // bias of hyperplane
        avx_vecAdd(centroid1.data(), centroid2.data(), centroid1.data(), centroid1.size());
        avx_vecScale(centroid1.data(), centroid1.data(), centroid1.size(), 0.5);
        return - avx_dotProduct(hyperplane.data(), centroid1.data(), centroid1.size());
    }
    
    // some math formula about geometry
    inline float Euclidean_distance(const Point& x, const Point& y) {
        return avx_L2Norm(x.data(), x.size())
               + avx_L2Norm(y.data(), y.size())
               - 2.0f * avx_dotProduct(x.data(), y.data(), y.size());
    }
    
    inline float margin(const Point& x, const Point& y, float bias) {
        return bias + avx_dotProduct(x.data(), y.data(), x.size());
    }
    
    inline bool side(const Point& x, const Point& y, float bias) {
        float dot = margin(x, y, bias);
        if (dot != 0)
            return (dot > 0); // margin > 0 go left
        else
            return SampleBinary(0.5f);
    }
    
    static const size_t leaf_min_points = 10;
    size_t feature_cnt;
    vector<Point> points_set;
    vector<Tree*> forest;
};

#endif /* ann_index_h */
