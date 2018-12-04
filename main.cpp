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

#include "LightCTR/distribut/master.h"
#include "LightCTR/distribut/paramserver.h"
#include "LightCTR/distribut/worker.h"
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
    Master();
#elif defined PS
    ParamServer<Key, Value>();
#elif defined WORKER
    Distributed_Algo_Abst *train = new Distributed_Algo_Abst(
                                 "./data/train_sparse",
                                 /*epoch*/50);
//    train->UnitTest();
    train->Train();
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
#ifdef TEST_ANN
    ANNIndex* annIndex = new ANNIndex("./output/word_embedding.txt", 100);
    vector<size_t> result;
    // query nearest neighbor of word embedding "state"
    float query_input[100] =  {-0.215541,0.251575,-0.123793,-0.18618,0.394535,0.266759,-0.159331,0.0909755,0.0540601,-0.276212,0.380615,-0.0750995,-0.122459,0.0973589,0.169981,-0.342782,-0.0320617,-0.275038,0.579765,0.0414525,0.0775329,0.150825,0.482595,0.342002,0.0489829,-0.104399,0.352892,0.470403,-0.073506,-0.0415242,0.0239289,0.327784,0.13405,0.15727,-0.262377,0.0391232,0.0853101,0.0493847,0.47749,-0.549036,0.259656,-0.0657005,0.566566,-0.273963,-0.196387,-0.494518,0.143773,-0.175798,0.409012,0.246421,-0.326558,0.373128,-0.199175,-0.409402,-0.196671,0.264,-0.0510461,0.106293,0.336967,0.275339,-0.0805158,0.296924,-0.17147,-0.533959,-0.455968,-0.034138,-0.280526,0.662283,-0.233623,-0.0299424,0.170299,-0.181736,-0.180484,0.0795437,0.0958492,0.14166,-0.0649171,0.267815,0.0393757,-0.00221303,0.0444415,-0.338447,0.154752,0.409157,0.0893646,0.263286,0.0204169,-0.171885,-0.102385,0.391818,-0.012545,-0.0170537,0.0460637,-0.446041,-0.111145,0.0875019,-0.123241,-0.26915,-0.0139937,0.224774};
    
    float16_t output[100];
    float out[100];
    auto transformer = Float16();
    transformer.convert2Float16(query_input, output, 100);
    transformer.recover2Float32(output, out, 100);
    for (int i = 0; i < 10; i++) {
        printf("(%f, %f) ", query_input[i], out[i]);
    }
    
    vector<float> input(query_input, query_input + 100);
    annIndex->query(input, 50, result);
    
    // test QuantileCompress
    uint8_t code[100];
    float unzip[100];
    QuantileCompress<float, uint8_t> compress(QuantileType::NORMAL_DISTRIBUT, -1, 1);
    compress.compress(query_input, 100, code);
    compress.extract(code, 100, unzip);
    for (int i = 0; i < 10; i++) {
        printf("(%f, %f) ", query_input[i], unzip[i]);
    }
    
    // test PCA
    vector<double> input_pca(unzip, unzip + 100);
    Matrix* m = new Matrix(1, 100, 0);
    m->loadDataPtr(&input_pca);
    PCA* train = new PCA(/*lr*/0.1, /*maxIters*/5,
                         /*neuronsNum*/5, /*featureSize*/100);
    train->loadMatrix(m);
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

