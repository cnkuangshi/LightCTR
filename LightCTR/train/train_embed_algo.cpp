//
//  train_embed_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/17.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_embed_algo.h"
#include <string.h>
#include <stack>
#include "../common/avx.h"
#include "../util/product_quantizer.h"

#define FOR_D for (size_t i = 0; i < emb_dimension; i++)

void Train_Embed_Algo::init() {
    learning_rate = 0.05f;
    negSample_cnt = 12;
    
    treeArry = new Node[vocab_cnt * 2];
    size_t treeNode_cnt = 0;
    // build Huffman
    priority_queue<pair<int, int>, vector<pair<int, int> >, cmp> Q;
    Node *curNode;
    int index = (int)vocab_cnt - 1;
    for (; index >= 0; index--) {
        curNode = treeArry + treeNode_cnt;
        curNode->frequency = word_frequency[index];
        Q.push(make_pair(word_frequency[index], treeNode_cnt));
        treeNode_cnt++;
    }
    while (Q.size() > 1) {
        pair<int, int> p1 = Q.top();
        Q.pop();
        pair<int, int> p2 = Q.top();
        Q.pop();
        curNode = treeArry + treeNode_cnt;
        curNode->weight = new float[emb_dimension];
        // hierarchical softmax weight init with 0
        memset(curNode->weight, 0, sizeof(float) * emb_dimension);
        
        curNode->frequency = p1.first + p2.first;
        curNode->left = p1.second;
        curNode->right = p2.second;
        Q.push(make_pair(p1.first + p2.first, treeNode_cnt));
        treeNode_cnt++;
    }
    treeRoot = treeArry + Q.top().second;
    Q.pop();
    assert(vocab_cnt * 2 - 1 == treeNode_cnt);
    
    // generate Huffman code, less frequency first
    word_code.resize(vocab_cnt);
    stack<pair<int, string> > S;
    S.push(make_pair(treeRoot - treeArry, string()));
    while (!S.empty()) {
        pair<size_t, string> p = S.top();
        pair<size_t, string> p2 = S.top();
        S.pop();
        curNode = treeArry + p.first;
        if (curNode->left == -1) { // get leaf
            assert(p.first < vocab_cnt);
            word_code[vocab_cnt - 1 - p.first] = p.second;
            continue;
        }
        p.first = curNode->left;
        p.second.append(move("1"));
        S.push(p);
        p2.first = curNode->right;
        p2.second.append(move("0"));
        S.push(p2);
    }
}

void Train_Embed_Algo::Train() {
    size_t docid = 0;
    string line;
    while(!textStream.eof()){
        getline(textStream, line);
        size_t offset = textStream.tellg();
        if (line == "<TEXT>") {
            threadpool->addTask(bind(&Train_Embed_Algo::TrainDocument, this, docid++, offset));
        }
    }
    threadpool->wait();
    cout << "All " << docid << " docs are trained completely" << endl;
    
    // Normalization
    for (size_t wid = 0; wid < vocab_cnt; wid++) {
        float* wd_ptr = word_embedding.data() + wid * emb_dimension;
        auto norm = avx_L2Norm(wd_ptr, emb_dimension);
        avx_vecScale(wd_ptr, wd_ptr, emb_dimension, 1.0 / std::sqrt(norm));
    }
    
    saveModel();
}

void Train_Embed_Algo::TrainDocument(size_t docid, size_t offset) {
    ifstream textStream_thread;
    loadTextFile(&textStream_thread);
    textStream_thread.seekg(offset);
    
    char word[128];
    vector<size_t> doc_wordid_vec;
    doc_wordid_vec.reserve(256);
    // extract all words id in one document
    while (NextWord(&textStream_thread, word) > 0) {
        auto it = vocabTable.find(string(word));
        if (it == vocabTable.end()) {
            continue;
        }
        if (subsampling > 0) { // subsample high frequent words
            float freq = word_frequency[vocabTable[word]];
            float prob = (sqrt(freq / (subsampling * total_words_cnt)) + 1) *
                        (subsampling * total_words_cnt) / freq;
            
            if (UniformNumRand() > prob)
                continue;
        }
        doc_wordid_vec.emplace_back(it->second);
    }
    if(doc_wordid_vec.size() <= window_size * 2 + 1) {
        return;
    }
    // N-Gram CBOW continuous bag of word
    // TODO skip-gram impl
    // cbow hsoftmax and negative discriminant
    vector<float> ctx_sum, emb_delta;
    ctx_sum.resize(emb_dimension);
    emb_delta.resize(emb_dimension);
    float decay_alpha = learning_rate;
    
    // each document trains epoch times
    for (size_t ep = 0; ep < epoch; ep++) {
        decay_alpha *= 0.96f;
        decay_alpha = fmax(decay_alpha, 0.0001);
        
        float hiso_loss = 0.0, negsa_loss = 0.0;
        int length = (int)doc_wordid_vec.size();
        for (int word_index = 0; word_index < length; word_index++) {
            fill(ctx_sum.begin(), ctx_sum.end(), 0);
            fill(emb_delta.begin(), emb_delta.end(), 0);
            // sum of embedding vector in window
            const int first = max(0, word_index - (int)window_size);
            const int last = min(length, word_index + (int)window_size);
            for (int pos = first; pos < last; pos++) {
                if (pos == word_index) {
                    continue;
                }
                size_t wid = doc_wordid_vec[pos];
                const float* wd_ptr = word_embedding.data() + wid * emb_dimension;
                avx_vecAdd(ctx_sum.data(), wd_ptr, ctx_sum.data(), emb_dimension);
            }
            
            // train hierarchical softmax
            const size_t cur_wid = doc_wordid_vec[word_index];
            const Node* curNode = treeRoot;
            for (size_t c = 0; c < word_code[cur_wid].length(); c++) {
                int realdir = word_code[cur_wid][c] - '0';
                float preddir = avx_dotProduct(curNode->weight, ctx_sum.data(), emb_dimension);
                if(!(preddir > -30 && preddir < 30)) {
                    printf("-- warning hiso %zu-%zu preddir = %f\n", cur_wid, c, preddir);
                }
                preddir = sigmoid.forward(preddir);
                hiso_loss += (1 == realdir) ? -log(preddir) : -log(1.0 - preddir);
                // LR gradient to max Loglikelihood
                float gradient = decay_alpha * (realdir - preddir);
                avx_vecScalerAdd(emb_delta.data(), curNode->weight,
                                 emb_delta.data(), gradient, emb_dimension);
                avx_vecScalerAdd(curNode->weight, ctx_sum.data(),
                                 curNode->weight, gradient, emb_dimension);
            }
            
            // train negative discriminant
            size_t label, wid = 0;
            for (size_t itr = 0; itr < negSample_cnt + 1; itr++) {
                if (itr == 0) {
                    label = 1;
                    wid = cur_wid;
                } else {
                    label = 0;
                    do { // smaple negative word
                        wid = negSampleTable[rand() % negTable_size];
                    } while (wid == cur_wid);
                }
                float* nw_ptr = negWeight.data() + wid * emb_dimension;
                float preddir = avx_dotProduct(nw_ptr, ctx_sum.data(), emb_dimension);
                if(!(preddir > -30 && preddir < 30)) {
                    printf("-- warning negsa %zu preddir = %f\n", cur_wid, preddir);
                }
                preddir = sigmoid.forward(preddir);
                negsa_loss += (1 == label) ? -log(preddir) : -log(1.0 - preddir);
                // LR gradient to max Loglikelihood
                float gradient = decay_alpha * (label - preddir);
                avx_vecScalerAdd(emb_delta.data(), nw_ptr,
                                 emb_delta.data(), gradient, emb_dimension);
                avx_vecScalerAdd(nw_ptr, ctx_sum.data(), nw_ptr, gradient, emb_dimension);
            }
            
            // unsafe multi-thread update word embedding
            for (int pos = first; pos < last; pos++) {
                size_t wid = doc_wordid_vec[pos];
                float* wd_ptr = word_embedding.data() + wid * emb_dimension;
                avx_vecAdd(wd_ptr, emb_delta.data(), wd_ptr, emb_dimension);
            }
        }
        cout << "docid " << docid << " epoch " << ep << " has " << doc_wordid_vec.size()
                        << " words" << " loss1 = " << hiso_loss
                        << " loss2 = " << negsa_loss << endl;
    }
}

void Train_Embed_Algo::Quantization(size_t part_cnt, uint8_t cluster_cnt) {
    Product_quantizer<float, uint8_t> pq(emb_dimension, part_cnt, cluster_cnt);
    auto quantizated_codes = pq.train(word_embedding.data(), vocab_cnt);
    
    ofstream md("./output/quantized_embedding.txt");
    if(!md.is_open()){
        cout<<"save model open file error" << endl;
        exit(1);
    }
    for (size_t wid = 0; wid < vocab_cnt; wid++) {
        for (size_t i = 0; i < part_cnt; i++) {
            md << static_cast<int>(quantizated_codes[i][wid]) << " ";
        }
        md << endl;
    }
    md << endl;
    md.close();
}

void Train_Embed_Algo::EmbeddingCluster(const vector<int>& clustered, size_t cluster_cnt) {
    
    vector<vector<string> > topicSet;
    topicSet.resize(cluster_cnt);
    for (size_t wid = 0; wid < clustered.size(); wid++) {
        assert(clustered[wid] < cluster_cnt);
        topicSet[clustered[wid]].push_back(vocabString[wid]);
    }
    ofstream md("./output/word_cluster.txt");
    if(!md.is_open()){
        cout<<"save model open file error" << endl;
        exit(1);
    }
    for (size_t c = 0; c < cluster_cnt; c++) {
        md << "Cluster " << c << ":";
        for (auto it = topicSet[c].begin(); it != topicSet[c].end(); it++) {
            md << " " << *it;
        }
        md << endl;
    }
    md.close();
    for (size_t c = 0; c < cluster_cnt; c++) {
        topicSet[c].clear();
    }
    topicSet.clear();
}

void Train_Embed_Algo::saveModel() {
    ofstream md("./output/word_embedding.txt");
    if(!md.is_open()){
        cout<<"save model open file error" << endl;
        exit(1);
    }
    for (size_t wid = 0; wid < vocab_cnt; wid++) {
        FOR_D {
            md << word_embedding[wid * emb_dimension + i] << " ";
        }
        md << endl;
    }
    md << endl;
    md.close();
}
