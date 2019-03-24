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
#include "../common/avx.h"

void Train_TM_Algo::init() {
    latentVar.resize(doc_cnt * word_cnt * topic_cnt);
    
    topics_of_docs.resize(doc_cnt * topic_cnt);
    latent_word_sum.resize(doc_cnt * topic_cnt);
    
    FOR(docid, doc_cnt) {
        threadpool->addTask([&, docid]() {
            float sum_tmp = 0.0f;
            FOR(tid, topic_cnt) {
                // if initialized with average 1.0f / topic_cnt, all topics_of_docs will be 0.1
                float r = 1.0f + static_cast<float>(rand() % 128);
                sum_tmp += r;
                topics_of_docs[docid * topic_cnt + tid] = r;
            }
            float* ptr = topics_of_docs.data() + docid * topic_cnt;
            avx_vecScale(ptr, ptr, topic_cnt, 1.0 / sum_tmp);
        });
    }
    words_of_topics.resize(topic_cnt * word_cnt);
    latent_doc_sum.resize(word_cnt * topic_cnt);
    
    FOR(tid, topic_cnt) {
        threadpool->addTask([&, tid]() {
            float sum_tmp = 0.0f;
            FOR(wid, word_cnt) {
                // if initialized with average 1.0f / word_cnt, all words' topics can't change
                float r = 1.0f + static_cast<float>(rand() % 128);
                sum_tmp += r;
                words_of_topics[tid * word_cnt + wid] = r;
            }
            float* ptr = words_of_topics.data() + tid * word_cnt;
            avx_vecScale(ptr, ptr, topic_cnt, 1.0 / sum_tmp);
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

vector<float>* Train_TM_Algo::Train_EStep() {
    FOR(docid, doc_cnt) {
        threadpool->addTask([&, docid]() {
            FOR(wid, word_cnt) {
                if (dataSet[docid][wid] == 0)
                    continue;
                float* ptr = latentVar.data() + docid * word_cnt * topic_cnt + wid * topic_cnt;
                FOR(tid, topic_cnt) {
                    *(ptr + tid) = words_of_topics[tid * word_cnt + wid]
                                 * topics_of_docs[docid * topic_cnt + tid];
                }
                float topic_sum = avx_L1Norm(ptr, topic_cnt);
                assert(topic_sum > 0);
                avx_vecScale(ptr, ptr, topic_cnt, 1.0 / topic_sum);
            }
        });
    }
    threadpool->wait();
    
    // cache for M-Step
    FOR(wid, word_cnt) {
        threadpool->addTask([&, wid]() {
            FOR(tid, topic_cnt) {
                float sum_tmp = 0.0f;
                FOR(docid, doc_cnt) {
                    if (dataSet[docid][wid] == 0)
                        continue;
                    // sum of (latentVar multiply term's frequency)
                    sum_tmp += dataSet[docid][wid]
                            * latentVar[docid * word_cnt * topic_cnt + wid * topic_cnt + tid];
                }
                latent_doc_sum[wid * topic_cnt + tid] = sum_tmp;
            }
        });
    }
    FOR(docid, doc_cnt) {
        threadpool->addTask([&, docid]() {
            FOR(tid, topic_cnt) {
                float sum_tmp = 0.0f;
                FOR(wid, word_cnt) {
                    if (dataSet[docid][wid] == 0)
                        continue;
                    sum_tmp += dataSet[docid][wid]
                            * latentVar[docid * word_cnt * topic_cnt + wid * topic_cnt + tid];
                }
                latent_word_sum[docid * topic_cnt + tid] = sum_tmp;
            }
        });
    }
    FOR(tid, topic_cnt) {
        threadpool->addTask([&, tid]() {
            float sum_tmp = 0.0f;
            FOR(docid, doc_cnt) {
                FOR(wid, word_cnt) {
                    if (dataSet[docid][wid] == 0)
                        continue;
                    sum_tmp += dataSet[docid][wid]
                            * latentVar[docid * word_cnt * topic_cnt + wid * topic_cnt + tid];
                }
            }
            latent_word_doc_sum[tid] = sum_tmp;
        });
    }
    threadpool->wait();
    return &latentVar;
}

float Train_TM_Algo::Train_MStep(const vector<float>*) {
    FOR(docid, doc_cnt) {
        const float tmp = 1.0 / wordCnt_of_doc[docid];
        avx_vecScale(latent_word_sum.data() + docid * topic_cnt,
                     topics_of_docs.data() + docid * topic_cnt, topic_cnt, tmp);
    }
    FOR(tid, topic_cnt) {
        threadpool->addTask([&, tid]() {
            const float tmp = latent_word_doc_sum[tid];
            FOR(wid, word_cnt) {
                words_of_topics[tid * word_cnt + wid] = latent_doc_sum[wid * topic_cnt + tid] / tmp;
            }
        });
    }
    threadpool->wait();

    // compute log likelihood ELOB
    float LogLKH = 0.0f;
    FOR(docid, doc_cnt) {
        FOR(wid, word_cnt) {
            if (dataSet[docid][wid] > 0) {
                float sum_tmp = 0.0f, tmp;
                FOR(tid, topic_cnt) {
                    float t1 = words_of_topics[tid * word_cnt + wid];
                    float t2 = topics_of_docs[docid * topic_cnt + tid];
                    assert(t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1);
                    tmp = log(words_of_topics[tid * word_cnt + wid] + 1e-7)
                        + log(topics_of_docs[docid * topic_cnt + tid] + 1e-7);
                    assert(!isnan(tmp));
                    sum_tmp += tmp
                            * latentVar[docid * word_cnt * topic_cnt + wid * topic_cnt + tid];
                }
                sum_tmp *= dataSet[docid][wid];
                LogLKH += sum_tmp;
                assert(!isnan(LogLKH));
            }
        }
    }
    return LogLKH;
}

vector<int> Train_TM_Algo::Predict() {
    vector<int> ans = vector<int>();
    return ans;
}

void Train_TM_Algo::printArguments() {
    vector<vector<string> > topicSet;
    topicSet.resize(topic_cnt);
    
    FOR(wid, word_cnt) {
        int whichTopic = -1;
        float maxP = 0.0f;
        FOR(tid, topic_cnt) {
            float sum_tmp = 0.0f;
            FOR(docid, doc_cnt) {
                if (dataSet[docid][wid] == 0) {
                    continue;
                }
                sum_tmp += topics_of_docs[docid * topic_cnt + tid] / dataSet[docid][wid];
            }
            sum_tmp *= words_of_topics[tid * word_cnt + wid];
            if (sum_tmp > maxP) {
                maxP = sum_tmp, whichTopic = (int)tid;
            }
        }
        if (whichTopic == -1) {
//            cout << "word " << vocab[wid] << " not exist" << endl;
        } else {
            topicSet[whichTopic].emplace_back(vocab[wid]);
        }
    }
    ofstream md("./output/topic_class.txt");
    if(!md.is_open()){
        cout<<"save model open file error" << endl;
        exit(1);
    }
    FOR(tid, topic_cnt) {
        md << "Topic " << tid << ":";
        for (auto it = topicSet[tid].begin(); it != topicSet[tid].end(); it++) {
            md << " " << *it;
        }
        md << endl;
    }
}
