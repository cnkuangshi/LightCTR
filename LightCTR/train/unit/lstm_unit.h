//
//  lstm_unit.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/10/31.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef lstm_unit_h
#define lstm_unit_h

#include "../layer/layer_abst.h"
#include "../../util/matrix.h"
using namespace std;

#define INIT_MATRIX(x) x##_w=new Matrix(dimension, hidden_size); x##_w->randomInit();               \
                       x##_grad_w=new Matrix(dimension, hidden_size); x##_grad_w->zeroInit();       \
                       x##_h_w=new Matrix(hidden_size, hidden_size); x##_h_w->randomInit();         \
                       x##_h_grad_w=new Matrix(hidden_size, hidden_size); x##_h_grad_w->zeroInit(); \
                       x##_b=new Matrix(1, hidden_size); x##_b->randomInit();                       \
                       x##_grad_b=new Matrix(1, hidden_size); x##_grad_b->zeroInit();               \
                       updater_##x##_w.learnable_params_cnt(dimension * hidden_size);               \
                       updater_##x##_h_w.learnable_params_cnt(hidden_size * hidden_size);           \
                       updater_##x##_b.learnable_params_cnt(hidden_size);

#define DEL_MATRIX(x) delete x##_w;delete x##_grad_w;delete x##_h_w;   \
                      delete x##_h_grad_w;delete x##_b;delete x##_grad_b;

#define INIT_DLETA(x) x##_delta=new Matrix(1, hidden_size); x##_delta->zeroInit();
#define DEL_DLETA(x) delete x##_delta;

#define ZERO_GRAD(x)   x##_grad_w->zeroInit();     \
                       x##_h_grad_w->zeroInit();   \
                       x##_grad_b->zeroInit();

#define UPDATE(x) updater_##x##_b.update(0, hidden_size, x##_b->reference(), x##_grad_b->reference()); \
                  updater_##x##_w.update(0, dimension * hidden_size, x##_w->reference(), x##_grad_w->reference()); \
                  updater_##x##_h_w.update(0, hidden_size * hidden_size, x##_h_w->reference(), x##_h_grad_w->reference());

// Bidirectional Recurrent Cell impl by Long Short Term Memory
template <typename ActivationFunction>
class LSTM_Unit : public Layer_Base {
    
public:
    LSTM_Unit(size_t _input_dimension, size_t _hidden_size, size_t _recurrent_cnt):
    Layer_Base(NULL, _input_dimension, _hidden_size), dimension(_input_dimension),
        batch_size(_recurrent_cnt), hidden_size(_hidden_size) {

        this->activeFun = new Identity();
        
        cur_seqid = 0;
        cache = new Matrix(1, hidden_size);
        cache_bp = new Matrix(dimension, hidden_size);
        cache_h_bp = new Matrix(hidden_size, hidden_size);
        
        error_clip_threshold = 15;
        wrapper = new vector<Matrix*>();
        wrapper->resize(1);
        
        // init all weight and bias matrix, each one has 3 matrix and their grad
        INIT_MATRIX(fg);
        INIT_MATRIX(inp);
        INIT_MATRIX(info);
        INIT_MATRIX(oup);
        
        input.resize(batch_size);
        fg_gate.resize(batch_size);
        inp_gate.resize(batch_size);
        info.resize(batch_size);
        oup_gate.resize(batch_size);
        c_state.resize(batch_size);
        extra_c_state.resize(batch_size);
        c_state_act.resize(batch_size);
        h_output.resize(batch_size);
        
        // alloc memory for c_state_delta;
        c_state_delta.resize(batch_size);
        // init *oup_gate_delta, *fg_gate_delta, *inp_gate_delta, *input_act_delta;
        INIT_DLETA(oup_gate);
        INIT_DLETA(fg_gate);
        INIT_DLETA(inp_gate);
        INIT_DLETA(input_act);
        
        h_output_delta = NULL;
        next_h_output_delta = NULL;
        
        printf("LSTM Unit\n");
    }
    
    ~LSTM_Unit() {
        DEL_MATRIX(fg);
        DEL_MATRIX(inp);
        DEL_MATRIX(info);
        DEL_MATRIX(oup);
        input.clear();
        fg_gate.clear();
        inp_gate.clear();
        info.clear();
        oup_gate.clear();
        c_state.clear();
        extra_c_state.clear();
        c_state_act.clear();
        h_output.clear();
        DEL_DLETA(oup_gate);
        DEL_DLETA(fg_gate);
        DEL_DLETA(inp_gate);
        DEL_DLETA(input_act);
    }
    
    // Attention to LSTM Unit forward wouldn't auto pass data into next layer
    vector<double>* forward(vector<Matrix*>* const prevLOutputMatrix) {
        if (cur_seqid == batch_size) {
            cur_seqid = 0; // new batch input row
        }
        assert(prevLOutputMatrix && prevLOutputMatrix->size() == 1);
        Matrix* inputRow = prevLOutputMatrix->at(0);
        assert(inputRow->x_len == 1);
        assert(inputRow->y_len == dimension);
        
        input[cur_seqid] = inputRow;
        
        // Gates structure build shortcut connection between h_t and h_t-1, c_t and c_t-1
        // forget gate and do sigmoid
        calculate(inputRow, fg_w, fg_h_w, fg_b, &fg_gate, &sigmoid);
        // input gate ..
        calculate(inputRow, inp_w, inp_h_w, inp_b, &inp_gate, &sigmoid);
        // info and do tanh
        calculate(inputRow, info_w, info_h_w, info_b, &info, &inner_activeFun);
        // output gate ..
        calculate(inputRow, oup_w, oup_h_w, oup_b, &oup_gate, &sigmoid);
        
        // obtain extra info
        extra_c_state[cur_seqid] = info[cur_seqid]->copy(extra_c_state[cur_seqid])->dotProduct(inp_gate[cur_seqid]);
        // apply forget gate and add to extra info
        if (cur_seqid > 0) {
            c_state[cur_seqid] = c_state[cur_seqid - 1]->copy(c_state[cur_seqid])->dotProduct(fg_gate[cur_seqid])->add(extra_c_state[cur_seqid]);
        } else {
            c_state[cur_seqid] = extra_c_state[cur_seqid]->copy(c_state[cur_seqid]);
        }
        // apply ouput gate after do tanh
        cache = c_state[cur_seqid]->copy(cache);
        inner_activeFun.forward(cache->pointer());
        c_state_act[cur_seqid] = cache->copy(c_state_act[cur_seqid]);
        h_output[cur_seqid] = cache->copy(h_output[cur_seqid])->dotProduct(oup_gate[cur_seqid]);
        
        assert(cur_seqid < batch_size);
        cur_seqid++;
        
        return h_output[cur_seqid - 1]->pointer();
    }
    
    void backward(vector<Matrix*>* const outputDeltaMatrix) {
        assert(cur_seqid == batch_size);
        Matrix* outputDelta = outputDeltaMatrix->at(0);
        assert(outputDelta->x_len == h_output[cur_seqid - 1]->x_len);
        assert(outputDelta->y_len == h_output[cur_seqid - 1]->y_len);
        if (outputDeltaMatrix->size() > 1) {
            assert(outputDeltaMatrix->size() == batch_size);
            outputDelta = outputDeltaMatrix->at((int)cur_seqid - 1);
        } else {
            outputDelta = outputDeltaMatrix->at(0);
        }
        next_h_output_delta = outputDelta->copy(next_h_output_delta);
        if (h_output_delta == NULL) {
            h_output_delta = new Matrix(1, hidden_size);
        }
        
        // BPTT Back Propagation Through Time, like onion
        for (int seqid = (int)cur_seqid - 1; seqid >= 0; seqid--) {
            swap(h_output_delta, next_h_output_delta);
            next_h_output_delta->zeroInit();
            if (outputDeltaMatrix->size() > 1) {
                // add this time's delta
                next_h_output_delta->add(outputDeltaMatrix->at(seqid));
            }
            
            // grad clipping
            if (error_clip_threshold != 0) {
                h_output_delta->clipping(error_clip_threshold);
            }
//            h_output_delta->debugPrint(); // unfolding delta for debug
            
            { // output gate weight
                oup_gate_delta = h_output_delta->copy(oup_gate_delta)->dotProduct(c_state_act[seqid]);
                sigmoid.backward(oup_gate_delta->pointer(), oup_gate[seqid]->pointer(), oup_gate_delta->pointer());
                
                accumGrad(oup_grad_w, oup_gate_delta, input[seqid]);
                if (seqid > 0) {
                    accumGrad(oup_h_grad_w, oup_gate_delta, h_output[seqid - 1]);
                }
                accumGrad(oup_grad_b, oup_gate_delta); // bias
                // 1-- h_t-1
                if (seqid > 0) {
                    accumGrad(next_h_output_delta, oup_gate_delta, oup_h_w);
                }
            }
            
            { // delta of c_state in t
                // begin from h_output_delta
                if (seqid < (int)cur_seqid - 1) { // accumulate the last time c_state's delta and h_output's delta
                    assert(c_state_delta[seqid]);
                    cache = h_output_delta->copy(cache)->dotProduct(oup_gate[seqid]);
                    inner_activeFun.backward(cache->pointer(), c_state_act[seqid]->pointer(), cache->pointer());
                    c_state_delta[seqid]->add(cache);
                } else { // for the first time of bp, clear memory
                    c_state_delta[seqid] = h_output_delta->copy(c_state_delta[seqid])->dotProduct(oup_gate[seqid]);
                    inner_activeFun.backward(c_state_delta[seqid]->pointer(), c_state_act[seqid]->pointer(), c_state_delta[seqid]->pointer());
                }
                
                { // delta of c_state in t-1, forget gate weight and delta of extra_info
                    // begin from c_state_delta[seqid]
                    
                    if (seqid > 0) {
                        // clear prev-time memory
                        c_state_delta[seqid - 1] = c_state_delta[seqid]->copy(c_state_delta[seqid - 1])->dotProduct(fg_gate[seqid]);
                        fg_gate_delta = c_state_delta[seqid]->copy(fg_gate_delta)->dotProduct(c_state[seqid - 1]);
                        sigmoid.backward(fg_gate_delta->pointer(), fg_gate[seqid]->pointer(), fg_gate_delta->pointer());
                        
                        accumGrad(fg_grad_w, fg_gate_delta, input[seqid]);
                        accumGrad(fg_h_grad_w, fg_gate_delta, h_output[seqid - 1]);
                        accumGrad(fg_grad_b, fg_gate_delta); // bias
                        // 2-- h_t-1
                        accumGrad(next_h_output_delta, fg_gate_delta, fg_h_w);
                    }
                    
                    { // delta of extra info, input gate weight and delta of h_t-1
                        
                        // input gate weight
                        inp_gate_delta = c_state_delta[seqid]->copy(inp_gate_delta)->dotProduct(info[seqid]);
                        sigmoid.backward(inp_gate_delta->pointer(), inp_gate[seqid]->pointer(), inp_gate_delta->pointer());
                        
                        accumGrad(inp_grad_w, inp_gate_delta, input[seqid]);
                        if (seqid > 0) {
                            accumGrad(inp_h_grad_w, inp_gate_delta, h_output[seqid - 1]);
                        }
                        accumGrad(inp_grad_b, inp_gate_delta); // bias
                        // 3-- h_t-1
                        if (seqid > 0) {
                            accumGrad(next_h_output_delta, inp_gate_delta, inp_h_w);
                        }
                        
                        // delta of input_act transform
                        input_act_delta = c_state_delta[seqid]->copy(input_act_delta)->dotProduct(inp_gate[seqid]);
                        inner_activeFun.backward(input_act_delta->pointer(), info[seqid]->pointer(), input_act_delta->pointer());
                        // input gate weight
                        accumGrad(info_grad_w, input_act_delta, input[seqid]);
                        if (seqid > 0) {
                            accumGrad(info_h_grad_w, input_act_delta, h_output[seqid - 1]);
                        }
                        accumGrad(info_grad_b, input_act_delta); // bias
                        // 4-- h_t-1
                        if (seqid > 0) {
                            accumGrad(next_h_output_delta, input_act_delta, info_h_w);
                        }
                    } // end extra
                } // end c_state
            } // end h
        } // end BPTT
    }
    
    const vector<Matrix*>* output() { // get last ouput as context vector
        wrapper->at(0) = h_output[cur_seqid - 1];
        return wrapper;
    }
    vector<Matrix*>* seq_output() { // get rnn encoder output sequence for attention decoder
        assert(cur_seqid == batch_size);
        return &h_output;
    }
    
    void applyBatchGradient() {
        assert(cur_seqid == batch_size);
        cur_seqid = 0;
        
        UPDATE(fg);
        UPDATE(inp);
        UPDATE(info);
        UPDATE(oup);
        
        if (nextLayer) {
            nextLayer->applyBatchGradient();
        }
    }
    
private:
    void accumGrad(Matrix* base, Matrix* delta, Matrix* grad = NULL) {
        unique_lock<mutex> glock(this->lock);
        assert(delta);
        if (grad) {
            if (base->x_len == dimension) { // w DxH
                grad->transpose()->Multiply(cache_bp, delta);
                grad->transpose(); // recover
                base->add(cache_bp);
            } else if (base->x_len == hidden_size) { // w_h HxH
                grad->transpose()->Multiply(cache_h_bp, delta);
                grad->transpose(); // recover
                base->add(cache_h_bp);
            } else { // h 1xH
                delta->Multiply(cache, grad);
                base->add(cache);
            }
        } else {
            base->add(delta);
        }
    }
    
    void calculate(Matrix* inputRow, Matrix* weight, Matrix* weight_h, Matrix* bias, vector<Matrix*>* target, Activation* actFun) {
        // input (1xD) * W_x (DxH) + H_t-1 (1xH) * W_h (HxH) + b (1xH)
        target->at(cur_seqid) = inputRow->Multiply(target->at(cur_seqid), weight);
        if (cur_seqid > 0) {
            cache = h_output[cur_seqid - 1]->Multiply(cache, weight_h);
            target->at(cur_seqid)->add(cache);
        }
        target->at(cur_seqid)->add(bias);
        actFun->forward(target->at(cur_seqid)->pointer());
    }
    
    size_t cur_seqid;
    size_t dimension, hidden_size;
    size_t batch_size;
    
    Sigmoid sigmoid;
    ActivationFunction inner_activeFun;
    // 8 weight
    Matrix *fg_w, *inp_w, *info_w, *oup_w;
    Matrix *fg_h_w, *inp_h_w, *info_h_w, *oup_h_w;
    // 4 bias
    Matrix *fg_b, *inp_b, *oup_b, *info_b;
    
    // history records of seq
    vector<Matrix*> fg_gate, inp_gate, info, oup_gate;
    vector<Matrix*> input, h_output, c_state, c_state_act, extra_c_state;
    
    // delta and gradient, resize when init
    vector<Matrix*> c_state_delta;
    // 8 weight grad
    Matrix *fg_grad_w, *inp_grad_w, *info_grad_w, *oup_grad_w;
    Matrix *fg_h_grad_w, *inp_h_grad_w, *info_h_grad_w, *oup_h_grad_w;
    // 4 bias grad
    Matrix *fg_grad_b, *inp_grad_b, *oup_grad_b, *info_grad_b;
    
    // delta cache, alloc when init
    Matrix *oup_gate_delta, *fg_gate_delta, *inp_gate_delta, *input_act_delta;
    // delta, don't need alloc
    Matrix *h_output_delta, *next_h_output_delta;
    // calculate temp matrix cache
    Matrix *cache, *cache_bp, *cache_h_bp;
    
    double error_clip_threshold;
    vector<Matrix*> *wrapper;
    
    // for updater
    AdagradUpdater_Num updater_fg_w, updater_fg_h_w, updater_fg_b;
    AdagradUpdater_Num updater_inp_w, updater_inp_h_w, updater_inp_b;
    AdagradUpdater_Num updater_info_w, updater_info_h_w, updater_info_b;
    AdagradUpdater_Num updater_oup_w, updater_oup_h_w, updater_oup_b;
};
#endif /* lstm_unit_h */
