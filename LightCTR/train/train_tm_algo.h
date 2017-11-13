//
//  train_tm_algo.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/15.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef train_tm_algo_h
#define train_tm_algo_h

#include <stdio.h>
#include <string>
#include "../common/thread_pool.h"
#include "../em_algo_abst.h"
using namespace std;

#define FOR(i,n) for(size_t i = 0;i < n;i++)

#define PLSA

// Topic Model impl by PLSA and Latent Dirichlet Allocation Algorithm
class Train_TM_Algo : public EM_Algo_Abst<vector<vector<double>* > > {
public:
    Train_TM_Algo(string _dataFile, string _vocabFile, size_t _epoch,
                  size_t _topic, size_t _words):
    EM_Algo_Abst(_dataFile, _epoch, _words), word_cnt(_words), topic_cnt(_topic) {
        doc_cnt = this->dataRow_cnt;
        threadpool = new ThreadPool(thread::hardware_concurrency());
        init();
        loadVocab(_vocabFile);
    }
    ~Train_TM_Algo() {
#ifdef PLSA
        FOR(docid, doc_cnt) {
            FOR(wid, word_cnt) {
                this->latentVar[docid]->at(wid)->clear();
                delete [] this->latentVar[docid]->at(wid);
            }
            delete [] this->latentVar[docid];
        }
        FOR(docid, doc_cnt) {
            topics_of_docs[docid].clear();
        }
        delete [] topics_of_docs;
        FOR(tid, topic_cnt) {
            words_of_topics[tid].clear();
        }
        delete [] words_of_topics;
        wordCnt_of_doc.clear();
        FOR(docid, doc_cnt) {
            latent_word_sum[docid].clear();
        }
        delete [] latent_word_sum;
        FOR(wid, word_cnt) {
            latent_doc_sum[wid].clear();
        }
        delete [] latent_doc_sum;
        latent_word_doc_sum.clear();
#endif
    }
    
    void init();
    vector<vector<double>* >** Train_EStep();
    double Train_MStep(vector<vector<double>* >**);
    
    void printArguments();
    shared_ptr<vector<int> > Predict();
    
    size_t word_cnt, topic_cnt, doc_cnt;
    vector<string> vocab;
    
    void loadVocab(string dataPath) {
        ifstream fin_;
        string line;
        char str[128];
        int val, fre;
        fin_.open(dataPath, ios::in);
        if(!fin_.is_open()){
            cout << "open file error, please run data/proc_text_topic.py first." << endl;
            exit(1);
        }
        while(!fin_.eof()){
            getline(fin_, line);
            const char *pline = line.c_str();
            if(sscanf(pline, "%d %s %d", &val, str, &fre) >= 1){
                assert(!isnan(val));
                vocab.push_back(string(str));
            }
        }
        assert(vocab.size() == word_cnt);
    }
    
    ThreadPool *threadpool;
    
#ifdef PLSA
    vector<vector<double>* >* *latentVar;
    vector<double>* topics_of_docs;
    vector<double>* words_of_topics;
    vector<size_t> wordCnt_of_doc;
    // cache for algorithm
    vector<double>* latent_word_sum; // word_sum[docid][tid] sum of all words
    vector<double>* latent_doc_sum; // doc_sum[wid][tid] sum of all docs
    vector<double> latent_word_doc_sum; // word_doc_sum[tid] sum of all docs and words
#endif
};

#endif /* train_tm_algo_h */
