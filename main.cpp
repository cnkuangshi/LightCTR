//
//  main.cpp
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/23.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#include <iostream>
#include "LightCTR/common/time.h"
#include "LightCTR/common/system.h"
#include "LightCTR/common/float16.h"
#include "LightCTR/util/pca.h"
#include "LightCTR/common/persistent_buffer.h"
#include "LightCTR/util/shm_hashtable.h"

#include "LightCTR/distribut/master.h"
#include "LightCTR/distribut/paramserver.h"
#include "LightCTR/distribut/dist_machine_abst.h"
#include "LightCTR/distribut/worker.h"
#include "LightCTR/distribut/ring_collect.h"
#include "LightCTR/distributed_algo_abst.h"

#include "LightCTR/fm_algo_abst.h"
#include "LightCTR/train/train_fm_algo.h"
#include "LightCTR/train/train_ffm_algo.h"
#include "LightCTR/train/train_nfm_algo.h"
#include "LightCTR/predict/fm_predict.h"

#include "LightCTR/gbm_algo_abst.h"
#include "LightCTR/train/train_gbm_algo.h"
#include "LightCTR/predict/gbm_predict.h"

#include "LightCTR/em_algo_abst.h"
#include "LightCTR/train/train_gmm_algo.h"
#include "LightCTR/train/train_tm_algo.h"

#include "LightCTR/train/train_embed_algo.h"

#include "LightCTR/dl_algo_abst.h"
#include "LightCTR/train/train_cnn_algo.h"
#include "LightCTR/train/train_rnn_algo.h"

#include "LightCTR/train/train_vae_algo.h"

#include "LightCTR/util/quantile_compress.h"
#include "LightCTR/util/product_quantizer.h"
#include "LightCTR/util/ensembling.h"
#include "LightCTR/predict/ann_index.h"
using namespace std;

// Attention to check config in GradientUpdater
//#define TEST_NFM

/* Recommend Configuration
 * Distributed LR lr=0.1
 * FM/FFM/NFM batch=50 lr=0.1
 * VAE batch=10 lr=0.1
 * CNN batch=10 lr=0.1
 * RNN batch=10 lr=0.03
 */

size_t GradientUpdater::__global_minibatch_size(50);
double GradientUpdater::__global_learning_rate(0.1);
double GradientUpdater::__global_ema_rate(0.99);
double GradientUpdater::__global_sparse_rate(0.6);
double GradientUpdater::__global_lambdaL2(0.001f);
double GradientUpdater::__global_lambdaL1(1e-5);
double MomentumUpdater::__global_momentum(0.8);
double MomentumUpdater::__global_momentum_adam2(0.999);

bool GradientUpdater::__global_bTraining(true);

int main(int argc, const char * argv[]) {
    
#ifdef MASTER
    {
        Master master(Run_Mode::PS_Mode);
    }
#elif defined MASTER_RING
    {
        Master master(Run_Mode::Ring_Mode);
    }
#elif defined PS
    {
        ParamServer<Key, Value>();
    }
#elif defined WORKER
    {
        Distributed_Algo_Abst *train = new Distributed_Algo_Abst(
                                     "./data/train_sparse",
                                     /*epoch*/50);
        train->Train();
    }
#elif defined WORKER_RING
    {
        puts("Run in Ring Mode");
        const int s = 50000;
        Worker_RingReduce<float> syncer(s, __global_cluster_worker_cnt);
        int Epoch = 20;
        for (size_t i = 0; i < Epoch; i++) {
            float grad[s];
            for (size_t j = 0; j < s; j++) {
                grad[j] = 1;
            }
            syncer.syncGradient(i, &grad[0]);
            for (size_t j = 0; j < s; j++) {
                assert(fabs(grad[j] - __global_cluster_worker_cnt) < 1e-5);
            }
            printf("**** Epoch %zu completed ****\n", i);
        }
        syncer.shutdown(NULL);
    }
#elif (defined TEST_FM) || (defined TEST_FFM) || (defined TEST_NFM) || (defined TEST_GBM) || (defined TEST_GMM) || (defined TEST_TM) || (defined TEST_EMB) || (defined TEST_CNN) || (defined TEST_RNN) || (defined TEST_VAE) || (defined TEST_ANN)
    int T = 200;
    
#ifdef TEST_FM
    FM_Algo_Abst *train = new Train_FM_Algo(
                        "./data/train_sparse.csv",
                        /*epoch*/3,
                        /*factor_cnt*/10);
    FM_Predict pred(train, "./data/train_sparse.csv", true);
#elif defined TEST_FFM
    FM_Algo_Abst *train = new Train_FFM_Algo(
                                            "./data/train_sparse.csv",
                                            /*epoch*/1,
                                            /*factor_cnt*/4,
                                            /*field*/68);
    FM_Predict pred(train, "./data/train_sparse.csv", true);
#elif defined TEST_NFM
    FM_Algo_Abst *train = new Train_NFM_Algo(
                                             "./data/train_sparse.csv",
                                             /*epoch*/3,
                                             /*factor_cnt*/10,
                                             /*hidden_layer_size*/32);
    FM_Predict pred(train, "./data/train_sparse.csv", true);
#elif defined TEST_GBM
    GBM_Algo_Abst *train = new Train_GBM_Algo(
                          "./data/train_dense.csv",
                          /*epoch*/1,
                          /*maxDepth*/12,
                          /*minLeafHess*/1,
                          /*multiclass*/10);
    GBM_Predict pred(train, "./data/train_dense.csv", true);
#elif defined TEST_GMM
    EM_Algo_Abst<vector<double> > *train =
    new Train_GMM_Algo(
                      "./data/train_cluster.csv",
                      /*epoch*/50, /*cluster_cnt*/100,
                      /*feature_cnt*/10);
    T = 1;
#elif defined TEST_TM
    EM_Algo_Abst<vector<vector<double>* > > *train =
    new Train_TM_Algo(
                   "./data/train_topic.csv",
                   "./data/vocab.txt",
                   /*epoch*/50,
                   /*topic*/5,
                   /*word*/5000);
    T = 1;
#elif defined TEST_EMB
    Train_Embed_Algo *train =
    new Train_Embed_Algo(
                       "./data/vocab.txt",
                       "./data/train_text.txt",
                       /*epoch*/50,
                       /*window_size*/6,
                       /*emb_dimention*/100,
                       /*vocab_cnt*/5000);
    T = 1;
#elif defined TEST_CNN
    DL_Algo_Abst<Square<double, int>, Tanh, Softmax> *train =
    new Train_CNN_Algo<Square<double, int>, Tanh, Softmax>(
                         "./data/train_dense.csv",
                         /*epoch*/300,
                         /*feature_cnt*/784,
                         /*hidden_size*/50,
                         /*multiclass_output_cnt*/10);
    T = 1;
#elif defined TEST_VAE
    Train_VAE_Algo<Square<double, double>, Sigmoid> *train =
    new Train_VAE_Algo<Square<double, double>, Sigmoid>(
                         "./data/train_dense.csv",
                         /*epoch*/600,
                         /*feature_cnt*/784,
                         /*hidden*/60,
                         /*gauss*/20);
    T = 1;
#elif defined TEST_RNN
    DL_Algo_Abst<Square<double, int>, Tanh, Softmax> *train =
    new Train_RNN_Algo<Square<double, int>, Tanh, Softmax>(
                         "./data/train_dense.csv",
                         /*epoch*/600,
                         /*feature_cnt*/784,
                         /*hidden_size*/50,
                         /*recurrent_cnt*/28,
                         /*multiclass_output_cnt*/10);
    T = 1;
#endif

    while (T--) {
        train->Train();
        
#if (defined TEST_FM) || (defined TEST_FFM) || (defined TEST_NFM) || (defined TEST_GBM)
        // Notice whether the algorithm have Predictor, otherwise Annotate it.
        pred.Predict("");
#endif
#ifdef TEST_EMB
//        train->loadPretrainFile("./output/word_embedding.txt");
        // Notice, word embedding vector multiply 10 to cluster
        EM_Algo_Abst<vector<double> > *cluster =
        new Train_GMM_Algo(
                        "./output/word_embedding.txt",
                        50,
                        50,
                        100,
                        /*scale*/10);
        train->Quantization(/*part_cnt*/20, /*cluster_cnt*/64);
        cluster->Train();
        shared_ptr<vector<int> > ans = cluster->Predict();
        train->EmbeddingCluster(ans, 50);
#endif
        cout << "------------" << endl;
    }
    train->saveModel(0);
    delete train;
#else
    puts("                         .'.                                     \n" \
         "                        ;'.',.                .'''.              \n" \
         "                        ;....''.            .,'...,'             \n" \
         "                        ;.......'''',''....''......;             \n" \
         "                     ..'........................;;.''            \n" \
         "                   ',...........................,...,            \n" \
         "                 ',...   ...........................;            \n" \
         "                ,'..;:c ......    ..................,.           \n" \
         "               ,'...oKK'......,lcOd'.................;           \n" \
         "              '.....',.......'K;oNNO.................''          \n" \
         "             ..   .','.........:ll;...................;          \n" \
         "           .',.  x0O0Kk,                             ...         \n" \
         "           .;.   ckOO0x,                               ;         \n" \
         "           .'    ..,,..                                '.        \n" \
         "           .'    .ckOxc'.   .                          ,         \n" \
         "            ;.      .';;;;'..                         .,         \n" \
         "            .'                 .                     .,          \n" \
         "             .'.    ...........                    .'.           \n" \
         "               '.     .......                    .'.             \n" \
         "                 .'.                         ....                \n" \
         "                   .......            .......                    \n" \
         "                         ..............                          \n\n\n");
    puts("Please define one algorithm to test, such as \n   [TEST_FM TEST_FFM TEST_NFM] \n" \
         "or [TEST_GBM TEST_GMM TEST_TM TEST_EMB] \n" \
         "or [TEST_CNN TEST_RNN TEST_VAE TEST_ANN]\n" \
         "or Different roles of cluster like [MASTER PS WORKER]\n");
#endif
    puts("Exit 0");
    return 0;
}

