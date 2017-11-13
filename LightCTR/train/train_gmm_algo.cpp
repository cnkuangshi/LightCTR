//
//  train_gmm_algo.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/13.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include "train_gmm_algo.h"
#include "../util/random.h"

#define FOR(i,n) for(size_t i = 0;i < n;i++)
const double PI = acos(-1);
const double LogPI = log(2 * PI);

inline double log_sum(double a, double b) {
    if (a < b) {
        return b + log(1.0f + exp(a - b));
    } else {
        return a + log(1.0f + exp(b - a));
    }
}

void Train_GMM_Algo::init() {
    this->gaussModels = new Gauss[cluster_cnt];
    this->latentVar = new vector<double>*[dataRow_cnt];
    FOR(i,cluster_cnt) {
        gaussModels[i].mu = new double[feature_cnt];
        gaussModels[i].sigma = new double[feature_cnt];
        memset(gaussModels[i].mu, 0, sizeof(double) * feature_cnt);
        FOR(fid, feature_cnt) {
            gaussModels[i].mu[fid] = (UniformNumRand() - 0.5f) * scale;
            gaussModels[i].sigma[fid] = 1.0f;
        }
        gaussModels[i].weight = 1.0f / cluster_cnt;
    }
    FOR(rid, dataRow_cnt) {
        latentVar[rid] = new vector<double>();
        latentVar[rid]->resize(cluster_cnt);
    }
}

// Log Probability Density Function of Multivariate Gauss Distribution
double Train_GMM_Algo::GaussianLPDF(size_t gasid, size_t rid) {
    double expN = 0, LogDetSigma = 0.0f, tmp = 0;
    FOR(fid, feature_cnt) {
        tmp = dataSet[rid][fid] * scale - gaussModels[gasid].mu[fid];
        expN += tmp * tmp / gaussModels[gasid].sigma[fid];
        LogDetSigma += log(gaussModels[gasid].sigma[fid]);
    }
    assert(!isnan(expN) && !isnan(LogDetSigma));
    tmp = log(gaussModels[gasid].weight) - 0.5 * (expN + LogDetSigma + feature_cnt * LogPI);
    assert(tmp < 0);
    return tmp;
}

vector<double>** Train_GMM_Algo::Train_EStep() {
    FOR(rid,dataRow_cnt) {
        double LogSumPDF = 0;
        FOR(gasid,cluster_cnt) {
            gaussModels[gasid].pdf_tmp = GaussianLPDF(gasid, rid);
            assert(gaussModels[gasid].pdf_tmp < 0);
            if (gasid == 0) {
                LogSumPDF = gaussModels[gasid].pdf_tmp;
            } else {
                LogSumPDF = log_sum(LogSumPDF, gaussModels[gasid].pdf_tmp);
            }
        }
        FOR(gasid,cluster_cnt) {
            double tmp = gaussModels[gasid].pdf_tmp - LogSumPDF;
            assert(tmp <= 0);
            this->latentVar[rid]->at(gasid) = exp(tmp);
//            printf("----rid %zu in cluster %zu latent %lf\n", rid, gasid, tmp);
        }
        assert(this->latentVar[rid]->size() == cluster_cnt);
    }
    return this->latentVar;
}

double Train_GMM_Algo::Train_MStep(vector<double>** latentVar) {
    threadpool->init();
    FOR(gasid, cluster_cnt) {
        threadpool->addTask([&, gasid]() {
            double newWeight = 0;
            FOR(rid,dataRow_cnt) {
                newWeight += latentVar[rid]->at(gasid);
            }
            assert(newWeight > 0 && newWeight < dataRow_cnt);
            gaussModels[gasid].sumRid_tmp = newWeight;
            // update new gauss weight
            gaussModels[gasid].weight = newWeight / dataRow_cnt;
            memset(gaussModels[gasid].mu, 0, sizeof(double) * feature_cnt);
            memset(gaussModels[gasid].sigma, 0, sizeof(double) * feature_cnt);
        });
    }
    threadpool->join();
    
    threadpool->init();
    FOR(gasid, cluster_cnt) {
        threadpool->addTask([&, gasid]() {
            // update new gauss mu
            FOR(fid, feature_cnt) {
                double sum_tmp = 0.0f;
                FOR(rid, dataRow_cnt) {
                    double t = latentVar[rid]->at(gasid) * dataSet[rid][fid] * scale;
                    sum_tmp += t;
                }
                gaussModels[gasid].mu[fid] = sum_tmp / gaussModels[gasid].sumRid_tmp;
            }
            // update new gauss sigma
            FOR(fid, feature_cnt) {
                double sum_tmp = 0.0f;
                FOR(rid, dataRow_cnt) {
                    double t = dataSet[rid][fid] * scale - gaussModels[gasid].mu[fid];
                    sum_tmp += latentVar[rid]->at(gasid) * t * t;
                }
                gaussModels[gasid].sigma[fid] = sum_tmp / gaussModels[gasid].sumRid_tmp;
                if (gaussModels[gasid].sigma[fid] < 1) {
                    gaussModels[gasid].sigma[fid] = 1.0f; // avoid detSigma beyand precision
                }
            }
        });
    }
    threadpool->join();
    
    // compute log likelihood ELOB
    double sumLogPDF = 0.0f;
    FOR(rid, dataRow_cnt) {
        FOR(gasid, cluster_cnt) {
            sumLogPDF += GaussianLPDF(gasid, rid) * latentVar[rid]->at(gasid);
            assert(!isnan(sumLogPDF));
        }
    }
    return sumLogPDF;
}

shared_ptr<vector<int> > Train_GMM_Algo::Predict() {
    shared_ptr<vector<int> > ans = shared_ptr<vector<int> >(new vector<int>());
    ans->reserve(dataRow_cnt);
    FOR(rid,dataRow_cnt) {
        int whichTopic = -1;
        double maxP = 0.0f, tmp;
        FOR(gasid,cluster_cnt) {
            tmp = GaussianLPDF(gasid, rid);
            assert(tmp < 0);
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

