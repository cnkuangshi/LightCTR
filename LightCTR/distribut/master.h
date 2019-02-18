//
//  master.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef master_h
#define master_h

#include <unordered_map>
#include "../common/system.h"
#include "../common/network.h"
#include "../common/lock.h"
#include "../common/barrier.h"
#include "../common/thread_pool.h"
#include "../common/message_queue.h"
#include "dist_machine_abst.h"
#include <thread>
#include <list>

const uint32_t __global_cluster_ps_cnt = getEnv("LightCTR_PS_NUM", 0); // read from env
const uint32_t __global_cluster_worker_cnt = getEnv("LightCTR_WORKER_NUM", 3);

class Master {
public:
    explicit Master(Run_Mode _run_mode) : gDelivery(Delivery::Instance()), run_mode(_run_mode) {
        registered_ps_cnt = 0;
        registered_worker_cnt = BEGIN_ID_OF_WORKER;
        
        if (run_mode == Run_Mode::PS_Mode) {
            printf("[Master] waiting for %d ps and %d worker\n",
                   __global_cluster_ps_cnt, __global_cluster_worker_cnt);
        } else if (run_mode == Run_Mode::Ring_Mode) {
            printf("[Master] waiting for %d worker\n", __global_cluster_worker_cnt);
        } else {
            assert(false);
        }
        
        regist_handshake_handler();
        regist_shutdown_handler();
        
        ping_barrier.block();
        
        {
            train_end_barrier = new Barrier(
                        (int)registered_worker_cnt - BEGIN_ID_OF_WORKER);
            terminate_barrier = new Barrier(
                        (int)registered_ps_cnt);
            
            start_heartbeat_monitor();
            broadcast_topology();
            
            train_end_barrier->block();
        }
        puts("[Master] shutdown cluster");
        
        broadcast_fin_toPS();
        
        terminate_barrier->block();
        serving = false;
        gDelivery.shutdown();
    }
    
    ~Master() {
        delete train_end_barrier;
        train_end_barrier = NULL;
        delete terminate_barrier;
        terminate_barrier = NULL;
    }
    
private:
    Master() = delete;
    
    void regist_handshake_handler() {
        request_handler_t handshake_handler = [this](std::shared_ptr<PackageDescript> request,
                                             PackageDescript& response) {
            const size_t node_id = request->node_id;
            if (node_id != BEGIN_ID_OF_PS && node_id != BEGIN_ID_OF_WORKER) {
                printf("[Master] node_id = %zu is re-connecting\n", node_id);
            }
            Addr node_addr(request->content.cursor());
            
            rwlock.wlock();
            if (run_mode == Run_Mode::PS_Mode) {
                if (node_id >= BEGIN_ID_OF_WORKER) { // a worker is registing
                    registered_worker_cnt++;
                    worker_list.emplace_back(std::move(node_addr));
                    gDelivery.regist_router(registered_worker_cnt, std::move(node_addr));
                    updateHeartbeat(registered_worker_cnt);
                    
                    request->node_id = registered_worker_cnt;
                    response.content << registered_worker_cnt;
                } else {
                    registered_ps_cnt++;
                    ps_list.emplace_back(std::move(node_addr));
                    gDelivery.regist_router(registered_ps_cnt, std::move(node_addr));
                    updateHeartbeat(registered_ps_cnt);
                    
                    request->node_id = registered_ps_cnt;
                    response.content << registered_ps_cnt;
                }
            } else if (run_mode == Run_Mode::Ring_Mode) {
                registered_worker_cnt++;
                worker_list.emplace_back(std::move(node_addr));
                gDelivery.regist_router(registered_worker_cnt, std::move(node_addr));
                updateHeartbeat(registered_worker_cnt);
                
                request->node_id = registered_worker_cnt;
                response.content << registered_worker_cnt;
            }
            rwlock.unlock();
            
            printf("[Master] Complete Register\n");
            
            assert(registered_ps_cnt <= config_ps_cnt);
            assert(registered_worker_cnt <= config_worker_cnt);
            
            if (registered_ps_cnt == config_ps_cnt &&
                registered_worker_cnt == config_worker_cnt) {
                ping_barrier.unblock();
            }
        };
        gDelivery.regist_handler(REQUEST_HANDSHAKE, std::move(handshake_handler));
    }
    
    void regist_shutdown_handler() {
        request_handler_t fin_handler = [this](std::shared_ptr<PackageDescript> request,
                                                     PackageDescript& response) {
            const size_t node_id = request->node_id;
            assert(node_id >= BEGIN_ID_OF_WORKER);
            assert(node_id <= registered_worker_cnt);
            if(gDelivery.delete_router(node_id)) {
                train_end_barrier->unblock();
                printf("[Master] node_id = %zu has shutdown\n", node_id);
            }
        };
        gDelivery.regist_handler(REQUEST_FIN, std::move(fin_handler));
    }
    
    void broadcast_topology() {
        PackageDescript resp_desc(REQUEST_ACK);
        
        std::string addr_str;
        
        if (run_mode == Run_Mode::PS_Mode) {
            puts("[Master] Broadcast PS and Worker topology");
            assert(registered_ps_cnt > 0 && registered_worker_cnt > 0);
            for (size_t ps_id = 1; ps_id <= registered_ps_cnt; ps_id++) {
                addr_str = gDelivery.get_router(ps_id).toString();
                resp_desc.content.append(addr_str.c_str(),
                                         addr_str.length());
            }
            printf("[1] Broadcast PS topology %.*s\n",
                   (int)resp_desc.content.size(), resp_desc.content.buffer());
            
            for (size_t w_id = BEGIN_ID_OF_WORKER + 1; w_id <= registered_worker_cnt; w_id++) {
                gDelivery.send_async(resp_desc, w_id);
            }
            
            resp_desc.content.reset();
            for (size_t w_id = BEGIN_ID_OF_WORKER + 1; w_id <= registered_worker_cnt; w_id++) {
                addr_str = gDelivery.get_router(w_id).toString();
                resp_desc.content.append(addr_str.c_str(),
                                         addr_str.length());
            }
            printf("[2] Broadcast Worker topology %.*s\n",
                   (int)resp_desc.content.size(), resp_desc.content.buffer());
            
            for (size_t ps_id = 1; ps_id <= registered_ps_cnt; ps_id++) {
                gDelivery.send_async(resp_desc, ps_id);
            }
        } else {
            puts("[Master] Broadcast Ring Worker topology");
            assert(registered_ps_cnt == 0 && registered_worker_cnt > 0);
            for (size_t w_id = BEGIN_ID_OF_WORKER + 1; w_id <= registered_worker_cnt; w_id++) {
                addr_str = gDelivery.get_router(w_id).toString();
                resp_desc.content.append(addr_str.c_str(),
                                         addr_str.length());
            }
            for (size_t w_id = BEGIN_ID_OF_WORKER + 1; w_id <= registered_worker_cnt; w_id++) {
                gDelivery.send_async(resp_desc, w_id);
            }
        }
    }
    
    void broadcast_fin_toPS() {
        PackageDescript desc(REQUEST_FIN);
        desc.callback = [this](std::shared_ptr<PackageDescript> resp_package) {
            terminate_barrier->unblock();
        };
        for (size_t ps_id = 1; ps_id <= registered_ps_cnt; ps_id++) {
            gDelivery.send_async(desc, ps_id);
        }
    }
    
    void start_heartbeat_monitor() {
        const time_t interval_ms = 5000;
        
        rwlock_heartbeat.rlock();
        for (auto it = heartbeats.begin(); it != heartbeats.end(); it++) {
            auto event = MessageEventWrapper(SendType::Period, interval_ms,
                                             [this, it](MessageEventWrapper& event) {
                const size_t node_id = it->first;
                if (node_id == 0) {
                    return;
                }
#ifdef DEBUG
                printf("[HEARTBEAT] checking alive of node_id = %zu\n", node_id);
#endif
                int res = checkAlive(node_id);
                if (res == -1) {
                    // stop router the node
                    printf("[Error][HEARTBEAT] %zu dead\n", node_id);
                    event.send_type = SendType::Invalid;
                    gDelivery.delete_router(node_id);
                    return;
                } else if (res == 0) {
                    if (event.after_or_period_time_ms == interval_ms)
                        event.after_or_period_time_ms *= 2;
                } else {
                    event.after_or_period_time_ms = interval_ms;
                }
            
                PackageDescript desc(HEARTBEAT);
                desc.callback = [this, node_id](
                                                std::shared_ptr<PackageDescript> resp_package) {
                    assert(resp_package->node_id == node_id);
                    updateHeartbeat(node_id);
                };
                gDelivery.send_async(desc, node_id);
            });
            heartbeats_runloop.emplace(std::move(event));
        }
        rwlock_heartbeat.unlock();
    }
    
    void updateHeartbeat(size_t node_id) {
        time_t cur_time = time(NULL);
        rwlock_heartbeat.wlock();
        heartbeats[node_id] = cur_time;
        rwlock_heartbeat.unlock();
    }
    
    int checkAlive(size_t node_id) {
        time_t cur_time = time(NULL);
        rwlock_heartbeat.rlock();
        assert(0 != heartbeats.count(node_id));
        time_t last_time = heartbeats[node_id];
        rwlock_heartbeat.unlock();
        if (last_time + 20 <= cur_time) {
            return -1; // 20s timeout means dead, give up
        } else if (last_time + 10 <= cur_time) {
            return 0; // 10s timeout, try re-send
        }
        return 1;
    }
    
    Run_Mode run_mode;
    
    std::list<Addr> ps_list;
    std::list<Addr> worker_list;
    size_t registered_ps_cnt;
    const size_t config_ps_cnt = __global_cluster_ps_cnt;
    size_t registered_worker_cnt;
    const size_t config_worker_cnt =
                BEGIN_ID_OF_WORKER + __global_cluster_worker_cnt;
    
    bool serving = true;
    RWLock rwlock;
    RWLock rwlock_heartbeat;
    MessageQueueRunloop heartbeats_runloop;
    std::map<size_t, time_t> heartbeats;
    
    Barrier ping_barrier;
    Barrier* train_end_barrier;
    Barrier* terminate_barrier;
    
    Delivery& gDelivery;
};

#endif /* master_h */
