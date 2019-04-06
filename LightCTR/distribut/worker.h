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
    Push push_op = Push('N');
    Pull pull_op = Pull('N');
    // for dense model
    Push push_tensor_op = Push('T');
    Pull pull_tensor_op = Pull('T');
    
private:
    ConsistentHash& gConsistentHash;
};

#endif /* worker_h */
