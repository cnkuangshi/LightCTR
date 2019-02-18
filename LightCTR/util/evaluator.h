//
//  evaluator.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/11/10.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef evaluator_h
#define evaluator_h

#include <string.h>

struct EvalInfo {
    // true positives, judge label=1 rightly
    float TP;
    // true negatives, judge label=0 rightly
    float TN;
    // false positives
    float FP;
    // false negatives
    float FN;
    
    EvalInfo() : TP(0.0), TN(0.0), FP(0.0), FN(0.0) {}
};

inline static float Precision(float TP, float FP) {
    if (TP > 0.0 || FP > 0.0) {
        return TP / (TP + FP);
    } else {
        return 1.0;
    }
}

inline static float Recall(float TP, float FN) {
    if (TP > 0.0 || FN > 0.0) {
        return TP / (TP + FN);
    } else {
        return 1.0;
    }
}

inline static float F1Score(float precision, float recall) {
    if (precision > 0.0 || recall > 0.0) {
        return 2.0f * precision * recall / (precision + recall);
    } else {
        return 0;
    }
}

class AucEvaluator {
public:
    AucEvaluator() {
        PosNum = new int[kHashLen + 1];
        NegNum = new int[kHashLen + 1];
    }
    ~AucEvaluator() {
        delete [] PosNum;
        delete [] NegNum;
    }
    void init(const vector<float>* pCTR, const vector<int>* label) {
        assert(pCTR->size() == label->size());
        memset(PosNum, 0, sizeof(int) * (kHashLen + 1));
        memset(NegNum, 0, sizeof(int) * (kHashLen + 1));
        
        for (size_t i = 0; i < pCTR->size(); i++) {
            size_t index = pCTR->at(i) * kHashLen;
            if (label->at(i) == 1) { // Positive
                PosNum[index]++;
            } else {
                NegNum[index]++;
            }
        }
    }
    float Auc() {
        float totPos = 0.0, totNeg = 0.0;
        float totPosPrev = 0.0, totNegPrev = 0.0;
        float auc = 0.0;
        
        int64_t idx = kHashLen;
        while (idx >= 0) {
            totPosPrev = totPos;
            totNegPrev = totNeg;
            totPos += PosNum[idx];
            totNeg += NegNum[idx];
            auc += trapezoidArea(totNeg, totNegPrev, totPos, totPosPrev);
            --idx;
        }
        if (totPos > 0.0 && totNeg > 0.0) {
            return auc / totPos / totNeg;
        } else {
            return 0.0;
        }
    }
private:
    inline float trapezoidArea(float X1, float X2,
                                float Y1, float Y2) {
        return (X1 > X2 ? (X1 - X2) : (X2 - X1)) * (Y1 + Y2) / 2.0;
    }
    
    const size_t kHashLen = (1 << 24) - 1;
    int *PosNum, *NegNum;
};

#endif /* evaluator_h */
