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
#include "../common/system.h"

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
    ANNIndex(string dataPath, size_t _feature_cnt, size_t tree_cnt = 5) :
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
            printf("%lu\n", forest[i]->size());
        }
    }
    
    void query(const Point& input, size_t beamSearchK, vector<size_t>& result_indices) {
        assert(beamSearchK > leaf_min_points);
        
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
        float bias = 0.0f;
        split_twoPart(indices, hyperplane, bias);
        
        vector<size_t> leftSet, rightSet;
        for (size_t i = 0; i < indices.size(); i++) {
            size_t index = indices[i];
            if (side(hyperplane, points_set[index], bias)) {
                leftSet.emplace_back(index);
            } else {
                rightSet.emplace_back(index);
            }
        }
        while (leftSet.size() == 0 || rightSet.size() == 0) {
            leftSet.clear(), rightSet.clear();
            hyperplane.clear();
            bias = 0;
            // random split
            for (size_t i = 0; i < indices.size(); i++) {
                if (SampleBinary(0.5f)) {
                    leftSet.emplace_back(indices[i]);
                } else {
                    rightSet.emplace_back(indices[i]);
                }
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
    
    inline void split_twoPart(const vector<size_t>& indices, Point& hyperplane, float &bias) {
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
                for (int i = 0; i < centroid1.size(); i++)
                    centroid1[i] = (centroid1[i] * ic + points_set[index][i]) / (ic + 1);
                ic++;
            } else if (dis1 > dis2) {
                for (int i = 0; i < centroid2.size(); i++)
                    centroid2[i] = (centroid2[i] * jc + points_set[index][i]) / (jc + 1);
                jc++;
            } else {
                // do nothing
            }
        }
        
        for (int i = 0; i < centroid2.size(); i++)
            hyperplane[i] = centroid1[i] - centroid2[i];
        normalize(hyperplane); // normalize to avoid repeat calculate
        for (int i = 0; i < centroid2.size(); i++)
            bias -= hyperplane[i] * (centroid1[i] + centroid2[i]) / 2.0f;
    }
    
    // some math formula about geometry
    inline float Euclidean_distance(const Point& x, const Point& y) {
        assert(x.size() == y.size());
        return avx_dotProduct(&x[0], &x[0], x.size()) + avx_dotProduct(&y[0], &y[0], y.size())
            - 2.0f * avx_dotProduct(&x[0], &y[0], y.size());
    }
    
    inline void normalize(Point& point) {
        float norm = avx_dotProduct(&point[0], &point[0], point.size());
        assert(norm > 0);
        norm = sqrt(norm);
        for (int i = 0; i < point.size(); i++) {
            point[i] /= norm;
        }
    }
    
    inline float margin(const Point& x, const Point& y, float bias) {
        assert(x.size() == y.size());
        return bias + avx_dotProduct(&x[0], &y[0], x.size());
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
