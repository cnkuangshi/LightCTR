//
//  dist_machine_abst.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017 SongKuangshi. All rights reserved.
//

#ifndef dist_machine_abst_h
#define dist_machine_abst_h

#include "../common/network.h"
#include "../common/barrier.h"
#include "../common/lock.h"

enum Run_Mode {
    PS_Mode = 0,
    Ring_Mode
};

class Dist_Machine_Abst {
public:
    Dist_Machine_Abst() : gDelivery(Delivery::Instance()) {
        gDelivery.set_node_id(BEGIN_ID_OF_WORKER);
        regist_curNode_toMaster();
        regist_master_ack_handler();
        
        serving_barrier.block();
        status_serving = true;
    }
    
    virtual ~Dist_Machine_Abst() {
        shutdown(NULL);
    }
    
    virtual inline size_t Rank() const { // Worker Rank begin from 1
        assert(status_serving);
        return gDelivery.node_id() - BEGIN_ID_OF_WORKER;
    }
    
    virtual inline bool status() const {
        return status_serving;
    }
    
    virtual inline void shutdown(std::function<void()> terminate_callback) {
        if (!status_serving) {
            return;
        }
        send_FIN_toMaster(terminate_callback);
    }
    
private:
    void regist_curNode_toMaster() {
        PackageDescript desc(REQUEST_HANDSHAKE);
        const Addr& local_addr = gDelivery.local_addr();
        desc.content.append(local_addr.toString().c_str(), local_addr.toString().length());
        
        desc.callback = [this](std::shared_ptr<PackageDescript> resp_package) {
            size_t node_id;
            resp_package->content >> node_id;
            printf("[Worker] Complete Register cur_node_id = %zu\n", node_id);
            gDelivery.set_node_id(node_id);
            assert(gDelivery.node_id() >= BEGIN_ID_OF_WORKER);
        };
        gDelivery.send_async(desc, 0);
    }
    
    void regist_master_ack_handler() {
        request_handler_t ack_handler = [this](
                                               std::shared_ptr<PackageDescript> request,
                                               PackageDescript& response) {
#ifdef WORKER_RING
            size_t ps_id = BEGIN_ID_OF_WORKER + 1;
#else
            size_t ps_id = BEGIN_ID_OF_PS;
#endif
            while (!request->content.readEOF()) { // read keys needed by worker
                Addr ps_addr(request->content);
                printf("[Worker] Add ps_id = %zu router\n", ps_id);
                gDelivery.regist_router(ps_id++, std::move(ps_addr));
            }
            serving_barrier.unblock();
        };
        gDelivery.regist_handler(REQUEST_ACK, std::move(ack_handler));
    }
    
    void send_FIN_toMaster(std::function<void()> terminate_callback) {
        PackageDescript desc(REQUEST_FIN);
        desc.callback = [this, terminate_callback](
                                                   std::shared_ptr<PackageDescript> resp_package) {
            puts("[Worker] Fin is accepted");
            gDelivery.shutdown();
            if (terminate_callback) {
                terminate_callback();
            }
        };
        gDelivery.send_async(desc, 0);
    }
    
    bool status_serving{false};
    Barrier serving_barrier;
protected:
    Delivery& gDelivery;
};

#endif /* dist_machine_abst_h */
