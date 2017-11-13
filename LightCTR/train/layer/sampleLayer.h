//
//  sampleLayer.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/21.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef sampleLayer_h
#define sampleLayer_h

#include "fullyconnLayer.h"
#include "../../util/random.h"

template <typename ActivationFunction>
class Sample_Layer : public Layer_Base {
public:
    Sample_Layer(Layer_Base* _prevLayer, size_t _input_dimention):
    Layer_Base(_prevLayer, _input_dimention, _input_dimention >> 1) {
        assert((_input_dimention & 1) == 0);
        gauss_cnt = _input_dimention >> 1;
        noise = new double[gauss_cnt];
        FOR(i, gauss_cnt) {
            noise[i] = GaussRand(); // only generate noise for sampling init once
        }
        bEncoding = false;
        
        this->activeFun = new ActivationFunction();
        
        inner_scale = 1.0f;
        
        printf("Sample Layer\n");
    }
    ~Sample_Layer() {
        delete noise;
    }
    
    vector<double>* forward(vector<Matrix*>* prevLOutputMatrix) {
        vector<double>* prevLOutput = prevLOutputMatrix->at(0)->pointer();
        assert(prevLOutput->size() == this->input_dimention);
        
        // init ThreadLocal var
        Matrix*& output_act = *tl_output_act;
        if (output_act == NULL) {
            output_act = new Matrix(1, this->output_dimention);
        }
        
        gaussDelta = 0.0f;
//        cout << "-- forw";
        FOR(i, gauss_cnt) {
            // prev layer output is mu and log(sigma^2)
            double mu = prevLOutput->at(i);
            double logSigma2 = prevLOutput->at(i + gauss_cnt);
            
            // min[ 0.5 * sum( exp(log_Sigma^2) - (1 + log_Sigma^2) + mu^2 ) ]
            gaussDelta += exp(inner_scale * logSigma2) - (1 + logSigma2) + mu * mu;
            assert(!isinf(gaussDelta));
            
//            cout << " " << mu << "," << logSigma2;
            // standard deviation equal to exp(0.5 * logSigma2)
            *output_act->getEle(0, i) = exp(inner_scale * 0.5f * logSigma2) * noise[i] + mu;
            assert(!isinf(*output_act->getEle(0, i)));
        }
        gaussDelta *= 0.5f;
//        cout << endl << endl << "gaussDelta = " << gaussDelta << endl << endl;
        if (bEncoding) {
            return output_act->pointer();
        }
        // init threadlocal wrapper
        vector<Matrix*>*& wrapper = *tl_wrapper;
        if (wrapper == NULL) {
            wrapper = new vector<Matrix*>();
            wrapper->resize(1);
        }
        
        wrapper->at(0) = output_act;
        return this->nextLayer->forward(wrapper);
    }
    
    void backward(vector<Matrix*>* outputDeltaMatrix) {
        assert(this->prevLayer);
        vector<double>* outputDelta = outputDeltaMatrix->at(0)->pointer();
        assert(outputDelta->size() == this->output_dimention);
        vector<double>* prev_output_act = this->prevLayer->output()->at(0)->pointer();
        assert(prev_output_act->size() == this->input_dimention);
        
        // init ThreadLocal var
        Matrix*& input_delta = *tl_input_delta;
        if (input_delta == NULL) {
            input_delta = new Matrix(1, this->input_dimention);
        }
        
        double delta_scale = 0.006;
        
        FOR(i, gauss_cnt) {
            assert(!isnan(outputDelta->at(i)));
            auto muPtr = input_delta->getEle(0, i);
            auto sigmaPtr = input_delta->getEle(0, i + gauss_cnt);
            
            // Target Loss about mu and log(sigma^2)
            auto sigmaGrad = 0.5f * exp(inner_scale * 0.5f * prev_output_act->at(i + gauss_cnt)) * noise[i];
            *muPtr = outputDelta->at(i);
            *sigmaPtr = outputDelta->at(i) * sigmaGrad;
            assert(!isinf(*sigmaPtr));
            
            // update Gauss Parameters Loss close to Normal distribution
            *muPtr += prev_output_act->at(i);
            *sigmaPtr += exp(inner_scale * prev_output_act->at(i + gauss_cnt)) - 1.0f;
            
            *muPtr *= delta_scale;
            *sigmaPtr *= delta_scale;
            
            assert(!isinf(*sigmaPtr));
        }
        this->prevLayer->getActiveFun().backward(input_delta->pointer(), prev_output_act, input_delta->pointer());
        
        vector<Matrix*>*& wrapper = *tl_wrapper;
        assert(wrapper);
        wrapper->at(0) = input_delta;
        this->prevLayer->backward(wrapper);
    }
    
    const vector<Matrix*>* output() {
        Matrix*& output_act = *tl_output_act;
        assert(output_act);
        
        vector<Matrix*>*& wrapper = *tl_wrapper;
        assert(wrapper);
        wrapper->at(0) = output_act;
        return wrapper;
    }
    
    bool bEncoding; // mark for forward encode
    
private:
    ThreadLocal<vector<Matrix*>*> tl_wrapper;
    
    ThreadLocal<Matrix*> tl_output_act; // wx + b with activation
    ThreadLocal<Matrix*> tl_input_delta; // delta of prevLayer wx+b Z_(L-1)
    
    double inner_scale;
    
    double gaussDelta;
    double* noise;
    size_t gauss_cnt;
};

#endif /* sampleLayer_h */
