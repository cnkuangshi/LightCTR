//
//  worker.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef worker_h
#define worker_h

#include "consistent_hash.h"
#include "dist_machine_abst.h"
#include "push.h"
#include "pull.h"

template <typename TKey, typename TValue>
class Worker : public Dist_Machine_Abst {
public:
    Worker() : gConsistentHash(ConsistentHash::Instance()) {
    }
    
    ~Worker() {
        
    }
    // for sparse model
    Push<TKey, TValue> push_op = Push<TKey, TValue>('N');
    Pull<TKey, TValue> pull_op = Pull<TKey, TValue>('N');
    // for dense model
    Push<TKey, TValue> push_tensor_op = Push<TKey, TValue>('T');
    Pull<TKey, TValue> pull_tensor_op = Pull<TKey, TValue>('T');
    
private:
    ConsistentHash& gConsistentHash;
};

#endif /* worker_h */
