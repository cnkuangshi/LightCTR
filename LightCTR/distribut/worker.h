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
    
    Push<TKey, TValue> push_op;
    Pull<TKey, TValue> pull_op;
    
private:
    ConsistentHash& gConsistentHash;
};

#endif /* worker_h */
