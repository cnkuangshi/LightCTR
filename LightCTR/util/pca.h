//
//  pca.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/5/4.
//  Copyright © 2018年 SongKuangshi. All rights reserved.
//

#ifndef pca_h
#define pca_h

#include "matrix.h"

// Functions for principal component analysis
class PCA {
public:
    PCA(float _learning_rate, int _maxIters, int _neuronsNum, int _featureSize) {
        learning_rate = _learning_rate;
        maxIters = _maxIters;
        neuronsNum = _neuronsNum;
        featureSize = _featureSize;
        
        weightsTmp = new Matrix(featureSize, neuronsNum);
        // Initializing Random weights for the first iteration
        weights = new Matrix(featureSize, neuronsNum);
        weights->randomInit();
    }
    
    void train(Matrix* trainingData) {
        // PCA trained by Generalized Hebbian Neuron
        for (int epoch = 0; epoch < maxIters; epoch++)
        {
            output = trainingData->Multiply(output, weights);
            weights->copy(weightsTmp);
            
            for (int row = 0; row < output->x_len; row++) {
                // each sample data
                for (int nid = 0; nid < neuronsNum; nid++) {
                    for (int fid = 0; fid < featureSize; fid++) {
                        // update each weight
                        float sumTerm = getSum(row, nid, fid);
                        *weights->getEle(fid, nid) -= learning_rate * *output->getEle(row, nid)
                                    * (*trainingData->getEle(row, fid) - sumTerm);
                    }
                }
            }
            
            if (weights->checkConvergence(weightsTmp)) {
                // if convergence then stop training
                printf("convergence in %d epoch", epoch);
                return;
            }
        }
        printf("[WARNING] stop training in %d epoch", maxIters);
    }
    
    Matrix* reduceDimension(Matrix* input, size_t reserve_pc_cnt = 1) {
        size_t orig = weights->y_len;
        weights->y_len = reserve_pc_cnt;
        output = input->Multiply(output, weights);
        weights->y_len = orig;
        return output;
    }
    
    Matrix* remove_pc(Matrix* input, size_t remove_pc_cnt = 1) {
        // V = V - (V * U) * U^T
        size_t orig = weights->y_len;
        weights->y_len = remove_pc_cnt;
        Matrix* lowDimentionM = NULL;
        lowDimentionM = input->Multiply(lowDimentionM, weights);
        output = lowDimentionM->Multiply(output, weights->transpose());
        output->add(input, 1, -1);
        weights->y_len = orig;
        
        return output;
    }
    
private:
    float getSum(int row, int nid, int fid) {
        float sum = 0;
        for (int i = 0; i <= nid; i++)
            sum += *output->getEle(row, i) * *weightsTmp->getEle(fid, i);
        return sum;
    }
    
    float learning_rate;
    int maxIters;
    int neuronsNum, featureSize;
    
    Matrix* weights = NULL;
    Matrix* weightsTmp = NULL;
    Matrix* output = NULL;
};

#endif /* pca_h */
