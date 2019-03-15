//
//  train_gmm_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/13.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_gmm_algo.h"
#include <string.h>
#include "../util/random.h"
#include "../common/avx.h"

#define FOR(i,n) for(size_t i = 0;i < n;i++)
const float PI = acos(-1);
const float LogPI = log(2 * PI);

// log(exp(a) + exp(b))
inline float log_sum(float a, float b) {
    const float vmin = std::min(a, b);
    const float vmax = std::max(a, b);
    if (vmax > vmin + 30) {
        return vmax;
    } else {
        return vmax + std::log(1.0 + std::exp(vmin - vmax));
    }
}

void Train_GMM_Algo::init() {
    gaussModels = new Gauss[cluster_cnt];
    latentVar.resize(dataRow_cnt * cluster_cnt);
    FOR(i,cluster_cnt) {
        gaussModels[i].mu = new float[feature_cnt];
        gaussModels[i].sigma = new float[feature_cnt];
        memset(gaussModels[i].mu, 0, sizeof(float) * feature_cnt);
        FOR(fid, feature_cnt) {
            gaussModels[i].mu[fid] = UniformNumRand() - 0.5f;
            gaussModels[i].sigma[fid] = 5.0f;
        }
        gaussModels[i].weight = 1.0f / cluster_cnt;
    }
}

// Log Probability Density Function of Multivariate Gauss Distribution
float Train_GMM_Algo::GaussianLPDF(size_t gasid, size_t rid) {
    float expN = 0, LogDetSigma = 0.0f, tmp = 0;
    FOR(fid, feature_cnt) {
        tmp = dataSet[rid][fid] * scale - gaussModels[gasid].mu[fid];
        expN += tmp * tmp / gaussModels[gasid].sigma[fid];
        LogDetSigma += log(gaussModels[gasid].sigma[fid]);
    }
    assert(!isnan(expN) && !isinf(expN) && !isnan(LogDetSigma) && !isinf(LogDetSigma));
    tmp = log(gaussModels[gasid].weight) - 0.5 * (expN + LogDetSigma + feature_cnt * LogPI);
//    assert(tmp < 0);
    return tmp;
}

vector<float>* Train_GMM_Algo::Train_EStep() {
    FOR(rid,dataRow_cnt) {
        float LogSumPDF = 0;
        FOR(gasid,cluster_cnt) {
            gaussModels[gasid].pdf_tmp = GaussianLPDF(gasid, rid);
            if (gasid == 0) {
                LogSumPDF = gaussModels[gasid].pdf_tmp;
            } else {
                LogSumPDF = log_sum(LogSumPDF, gaussModels[gasid].pdf_tmp);
            }
        }
        // Normalization
        float expSum = 0;
        FOR(gasid,cluster_cnt) {
            float tmp = exp(gaussModels[gasid].pdf_tmp - LogSumPDF);
            assert(tmp <= 1);
            latentVar[rid * cluster_cnt + gasid] = tmp;
            expSum += tmp;
        }
        float* ptr = latentVar.data() + rid * cluster_cnt;
        avx_vecScale(ptr, ptr, cluster_cnt, 1.0 / expSum);
    }
    return &latentVar;
}

float Train_GMM_Algo::Train_MStep(const vector<float>* latentVar) {
    FOR(gasid, cluster_cnt) {
        threadpool->addTask([&, gasid]() {
            float sumWeight = 0;
            FOR(rid,dataRow_cnt) {
                sumWeight += latentVar->at(rid * cluster_cnt + gasid);
            }
            assert(sumWeight > 0 && sumWeight < dataRow_cnt);
            gaussModels[gasid].sumRid_tmp = sumWeight;
            // update new gauss weight
            gaussModels[gasid].weight = sumWeight / dataRow_cnt;
        });
    }
    threadpool->wait();
    
    FOR(gasid, cluster_cnt) {
        threadpool->addTask([&, gasid]() {
            auto model = gaussModels[gasid];
            // update new gauss mu and sigma
            FOR(fid, feature_cnt) {
                float sum_mu = 0.0f, sum_sigma = 0.0f;
                FOR(rid, dataRow_cnt) {
                    sum_mu += latentVar->at(rid * cluster_cnt + gasid) * dataSet[rid][fid] * scale;
                    const float t = dataSet[rid][fid] * scale - model.mu[fid];
                    sum_sigma += latentVar->at(rid * cluster_cnt + gasid) * t * t;
                }
                model.mu[fid] = sum_mu / model.sumRid_tmp;
                model.sigma[fid] = sum_sigma / model.sumRid_tmp;
                if (model.sigma[fid] < 0.01) {
                    model.sigma[fid] = 0.01; // avoid detSigma beyand precision
                }
            }
        });
    }
    threadpool->wait();
    
    // compute log likelihood ELOB
    float likelihood = 0.0f;
    FOR(rid,dataRow_cnt) {
        float tmp = 0.0, raw_log_sum = 0.0;
        FOR(gasid,cluster_cnt) {
            tmp = GaussianLPDF(gasid, rid);
            if (gasid == 0) {
                raw_log_sum = tmp;
            } else {
                raw_log_sum = log_sum(raw_log_sum, tmp);
            }
        }
        likelihood += raw_log_sum;
    }
    return likelihood;
}

shared_ptr<vector<int> > Train_GMM_Algo::Predict() {
    shared_ptr<vector<int> > ans = std::make_shared<vector<int> >();
    ans->reserve(dataRow_cnt);
    FOR(rid,dataRow_cnt) {
        int whichTopic = -1;
        float maxP = 0.0f, tmp;
        FOR(gasid,cluster_cnt) {
            tmp = GaussianLPDF(gasid, rid);
            if (whichTopic == -1 || tmp > maxP) {
                maxP = tmp, whichTopic = (int)gasid;
            }
        }
        ans->emplace_back(whichTopic);
    }
    return ans;
}

void Train_GMM_Algo::printArguments() {
    ofstream md("./output/gmm_cluster.txt");
    if(!md.is_open()){
        cout<<"save model open file error" << endl;
        exit(1);
    }
    FOR(gasid, cluster_cnt) {
        md << "cluster " << gasid << " weight =";
        md << " " << gaussModels[gasid].weight << endl;
        md << "cluster " << gasid << " mu =";
        FOR(fid, feature_cnt) {
            md << " " << gaussModels[gasid].mu[fid];
        }
        md << endl;
        md << "cluster " << gasid << " sigma =";
        FOR(fid, feature_cnt) {
            md << " " << gaussModels[gasid].sigma[fid];
        }
        md << endl;
    }
    md.close();
}

