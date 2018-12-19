//
//  train_tm_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/15.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_tm_algo.h"
#include <cmath>
#include <stdlib.h>

void Train_TM_Algo::init() {
    this->latentVar = new vector<vector<double>* >*[doc_cnt];
    
    FOR(docid, doc_cnt) {
        threadpool->addTask([&, docid]() {
            this->latentVar[docid] = new vector<vector<double>* >();
            this->latentVar[docid]->reserve(word_cnt);
            FOR(wid, word_cnt) {
                vector<double>* vec = new vector<double>();
                this->latentVar[docid]->emplace_back(vec);
                vec->resize(topic_cnt);
            }
            assert(this->latentVar[docid]->size() == word_cnt);
        });
    }
    threadpool->wait();
    topics_of_docs = new vector<double>[doc_cnt];
    latent_word_sum = new vector<double>[doc_cnt];
    
    FOR(docid, doc_cnt) {
        threadpool->addTask([&, docid]() {
            latent_word_sum[docid].resize(topic_cnt);
            
            topics_of_docs[docid].reserve(topic_cnt);
            double sum_tmp = 0.0f;
            FOR(tid, topic_cnt) {
                // if initialized with average 1.0f / topic_cnt, all topics_of_docs will be 0.1
                double r = 1.0f + static_cast<double>(rand() % 128);
                sum_tmp += r;
                topics_of_docs[docid].emplace_back(r);
            }
            FOR(tid, topic_cnt) {
                topics_of_docs[docid][tid] /= sum_tmp;
            }
        });
    }
    words_of_topics = new vector<double>[topic_cnt];
    latent_doc_sum = new vector<double>[word_cnt];
    FOR(wid, word_cnt) {
        latent_doc_sum[wid].resize(topic_cnt);
    }
    FOR(tid, topic_cnt) {
        threadpool->addTask([&, tid]() {
            words_of_topics[tid].reserve(word_cnt);
            double sum_tmp = 0.0f;
            FOR(wid, word_cnt) {
                // if initialized with average 1.0f / word_cnt, all words' topics can't change
                double r = 1.0f + static_cast<double>(rand() % 128);
                sum_tmp += r;
                words_of_topics[tid].emplace_back(r);
            }
            FOR(wid, word_cnt) {
                words_of_topics[tid][wid] /= sum_tmp;
            }
        });
    }
    threadpool->wait();
    
    latent_word_doc_sum.resize(topic_cnt);
    wordCnt_of_doc.resize(doc_cnt);
    FOR(docid, doc_cnt) {
        wordCnt_of_doc[docid] = 0;
        FOR(wid, word_cnt) {
            wordCnt_of_doc[docid] += dataSet[docid][wid];
        }
        assert(wordCnt_of_doc[docid] > 0);
    }
}

vector<vector<double>* >** Train_TM_Algo::Train_EStep() {
    FOR(docid, doc_cnt) {
        threadpool->addTask([&, docid]() {
            FOR(wid, word_cnt) {
                if (dataSet[docid][wid] == 0) {
                    continue;
                }
                double topic_sum = 0.0f, tmp;
                FOR(tid, topic_cnt) {
                    tmp = words_of_topics[tid][wid] * topics_of_docs[docid][tid];
                    latentVar[docid]->at(wid)->at(tid) = tmp;
                    topic_sum += tmp;
                }
                assert(topic_sum > 0);
                FOR(tid, topic_cnt) {
                    tmp = latentVar[docid]->at(wid)->at(tid) / topic_sum;
                    assert(tmp >= 0 && tmp <= 1);
                    latentVar[docid]->at(wid)->at(tid) = tmp;
                }
            }
        });
    }
    threadpool->wait();
    
    // cache for M-Step
    FOR(wid, word_cnt) {
        threadpool->addTask([&, wid]() {
            FOR(tid, topic_cnt) {
                double sum_tmp = 0.0f;
                bool f = 0;
                FOR(docid, doc_cnt) {
                    if (dataSet[docid][wid] > 0) {
                        f = 1;
                        // sum of (latentVar multiply term's frequency)
                        sum_tmp += latentVar[docid]->at(wid)->at(tid) * dataSet[docid][wid];
                    }
                }
                latent_doc_sum[wid][tid] = sum_tmp;
            }
        });
    }
    FOR(docid, doc_cnt) {
        threadpool->addTask([&, docid]() {
            FOR(tid, topic_cnt) {
                double sum_tmp = 0.0f;
                bool f = 0;
                FOR(wid, word_cnt) {
                    if (dataSet[docid][wid] > 0) {
                        f = 1;
                        sum_tmp += latentVar[docid]->at(wid)->at(tid) * dataSet[docid][wid];
                    }
                }
                latent_word_sum[docid][tid] = sum_tmp;
            }
        });
    }
    FOR(tid, topic_cnt) {
        threadpool->addTask([&, tid]() {
            double sum_tmp = 0.0f;
            bool f = 0;
            latent_word_doc_sum[tid] = 0;
            FOR(docid, doc_cnt) {
                FOR(wid, word_cnt) {
                    if (dataSet[docid][wid] > 0) {
                        f = 1;
                        sum_tmp += latentVar[docid]->at(wid)->at(tid) * dataSet[docid][wid];
                    }
                }
            }
            latent_word_doc_sum[tid] = sum_tmp;
        });
    }
    threadpool->wait();
    return latentVar;
}

double Train_TM_Algo::Train_MStep(vector<vector<double>* >**) {
    FOR(docid, doc_cnt) {
        threadpool->addTask([&, docid]() {
            double tmp;
            FOR(tid, topic_cnt) {
                assert(wordCnt_of_doc[docid] > 0);
                tmp = latent_word_sum[docid][tid] / wordCnt_of_doc[docid];
                assert(tmp >= 0 && tmp <= 1);
                topics_of_docs[docid][tid] = tmp;
            }
        });
    }
    FOR(tid, topic_cnt) {
        threadpool->addTask([&, tid]() {
            double tmp;
            FOR(wid, word_cnt) {
                assert(latent_word_doc_sum[tid] > 0);
                tmp = latent_doc_sum[wid][tid] / latent_word_doc_sum[tid];
                assert(tmp >= 0 && tmp <= 1);
                words_of_topics[tid][wid] = tmp;
            }
        });
    }
    threadpool->wait();
    
    // check
    FOR(tid, topic_cnt) {
        double sum = 0.0f, tmp;
        FOR(wid, word_cnt) {
            tmp = words_of_topics[tid][wid];
            sum += tmp;
        }
        assert(fabs(sum - 1.0f) < 1e-6);
    }
    FOR(docid, doc_cnt) {
        double sum = 0.0f, tmp;
        FOR(tid, topic_cnt) {
            tmp = topics_of_docs[docid][tid];
            sum += tmp;
        }
        assert(fabs(sum - 1.0f) < 1e-6);
    }
    
    // compute log likelihood ELOB
    double LogLKH = 0.0f;
    FOR(docid, doc_cnt) {
        FOR(wid, word_cnt) {
            if (dataSet[docid][wid] > 0) {
                double sum_tmp = 0.0f, tmp;
                FOR(tid, topic_cnt) {
                    double t1 = words_of_topics[tid][wid];
                    double t2 = topics_of_docs[docid][tid];
                    assert(t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1);
                    tmp = log(words_of_topics[tid][wid] + 1e-12) + log(topics_of_docs[docid][tid] + 1e-12);
                    assert(!isnan(tmp));
                    sum_tmp += latentVar[docid]->at(wid)->at(tid) * tmp;
                }
                sum_tmp *= dataSet[docid][wid];
                LogLKH += sum_tmp;
                assert(!isnan(LogLKH));
            }
        }
    }
    return LogLKH;
}

shared_ptr<vector<int> > Train_TM_Algo::Predict() {
    shared_ptr<vector<int> > ans = shared_ptr<vector<int> >(new vector<int>());
    return ans;
}

void Train_TM_Algo::printArguments() {
    vector<vector<string>* > topicSet;
    topicSet.reserve(topic_cnt);
    FOR(tid, topic_cnt) {
        topicSet.emplace_back(new vector<string>);
        topicSet[tid]->reserve(word_cnt / topic_cnt);
    }
    FOR(wid, word_cnt) {
        int whichTopic = -1;
        double maxP = 0.0f;
        FOR(tid, topic_cnt) {
            double sum_tmp = 0.0f;
            FOR(docid, doc_cnt) {
                if (dataSet[docid][wid] == 0) {
                    continue;
                }
                sum_tmp += topics_of_docs[docid][tid] * words_of_topics[tid][wid] / dataSet[docid][wid];
            }
            if (sum_tmp > maxP) {
                maxP = sum_tmp, whichTopic = (int)tid;
            }
        }
        if (whichTopic == -1) {
//            cout << "word " << vocab[wid] << " not exist" << endl;
        } else {
            topicSet[whichTopic]->emplace_back(vocab[wid]);
        }
    }
    FOR(tid, topic_cnt) {
        cout << "Topic " << tid << ":";
        for (auto it = topicSet[tid]->begin(); it != topicSet[tid]->end(); it++) {
            cout << " " << *it;
        }
        puts("");
    }
}
