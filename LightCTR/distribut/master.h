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
#include <thread>
#include <list>

const uint32_t __global_cluster_ps_cnt = getEnv("LightCTR_PS_NUM", 2); // read from env
const uint32_t __global_cluster_worker_cnt = getEnv("LightCTR_WORKER_NUM", 2);

class Master {
public:
    Master() : gDelivery(Delivery::Instance()) {
        registered_ps_cnt = 0;
        registered_worker_cnt = BEGIN_ID_OF_WORKER;
        
        printf("[Master] waiting for %d ps and %d worker\n",
               __global_cluster_ps_cnt, __global_cluster_worker_cnt);
        
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
        threadpool->join();
        gDelivery.shutdown();
    }
    
private:
    void regist_handshake_handler() {
        request_handler_t handshake_handler = [this](std::shared_ptr<PackageDescript> request,
                                             PackageDescript& response) {
            const size_t node_id = request->node_id;
            if (node_id != BEGIN_ID_OF_PS && node_id != BEGIN_ID_OF_WORKER) {
                printf("[Master] node_id = %zu is re-connecting\n", node_id);
            }
            Addr node_addr(request->content);
            
            rwlock.wlock();
            if (node_id >= BEGIN_ID_OF_WORKER) { // a worker is registing
                registered_worker_cnt++;
                worker_list.emplace_back(std::move(node_addr));
                gDelivery.regist_router(registered_worker_cnt, node_addr);
                updateHeartbeat(registered_worker_cnt);
                
                request->node_id = registered_worker_cnt;
                response.content << registered_worker_cnt;
            } else {
                registered_ps_cnt++;
                ps_list.emplace_back(std::move(node_addr));
                gDelivery.regist_router(registered_ps_cnt, node_addr);
                updateHeartbeat(registered_ps_cnt);
                
                request->node_id = registered_ps_cnt;
                response.content << registered_ps_cnt;
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
            train_end_barrier->unblock();
            
            printf("[Master] node_id = %zu has shutdown\n", node_id);
        };
        gDelivery.regist_handler(REQUEST_FIN, std::move(fin_handler));
    }
    
    void broadcast_topology() {
        PackageDescript resp_desc(REQUEST_ACK);
        
        std::string ps_addr_str;
        
        for (size_t ps_id = 1; ps_id <= registered_ps_cnt; ps_id++) {
            ps_addr_str = gDelivery.get_router(ps_id).toString();
            resp_desc.content.append(ps_addr_str.c_str(),
                                     ps_addr_str.length());
        }
        printf("[1] Broadcast PS topology %.*s\n",
               (int)resp_desc.content.size(), resp_desc.content.buffer());
        
        for (size_t w_id = BEGIN_ID_OF_WORKER + 1; w_id <= registered_worker_cnt; w_id++) {
            gDelivery.send_sync(std::move(resp_desc), w_id);
        }
        
        resp_desc.content.reset();
        for (size_t w_id = BEGIN_ID_OF_WORKER + 1; w_id <= registered_worker_cnt; w_id++) {
            ps_addr_str = gDelivery.get_router(w_id).toString();
            resp_desc.content.append(ps_addr_str.c_str(),
                                     ps_addr_str.length());
        }
        printf("[2] Broadcast Worker topology %.*s\n",
               (int)resp_desc.content.size(), resp_desc.content.buffer());
        
        for (size_t ps_id = 1; ps_id <= registered_ps_cnt; ps_id++) {
            gDelivery.send_sync(std::move(resp_desc), ps_id);
        }
    }
    
    void broadcast_fin_toPS() {
        PackageDescript desc(REQUEST_FIN);
        desc.callback = [this](std::shared_ptr<PackageDescript> resp_package) {
            terminate_barrier->unblock();
        };
        for (size_t ps_id = 1; ps_id <= registered_ps_cnt; ps_id++) {
            gDelivery.send_sync(std::move(desc), ps_id);
        }
    }
    
    void start_heartbeat_monitor() {
        const time_t interval = 10;
        threadpool = new ThreadPool(std::thread::hardware_concurrency());
        for (auto it = heartbeats.begin(); it != heartbeats.end(); it++) {
            const size_t node_id = it->first;
            if (node_id == 0) {
                continue;
            }
            threadpool->addTask([this, interval, node_id]() {
                while (serving && interval > 0) {
                    std::this_thread::sleep_for(std::chrono::seconds(interval));
                    int res = checkAlive(node_id);
                    if (res == 0) {
                        // stop router
                        gDelivery.delete_router(node_id);
                    } else if (res == -1) {
                        return; // the node is dead, stop heartbeat
                    }
                    printf("[HEARTBEAT] checking alive of node_id = %zu\n", node_id);
                    
                    PackageDescript desc(HEARTBEAT);
                    desc.callback = [this, node_id](
                            std::shared_ptr<PackageDescript> resp_package) {
                        assert(resp_package->node_id == node_id);
                        updateHeartbeat(node_id);
                    };
                    gDelivery.send_sync(std::move(desc), node_id);
                }
            });
        }
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
        if (last_time + 30 <= cur_time) {
            return 0; // 30s timeout
        } else if (last_time + 40 <= cur_time) {
            return -1; // 40s timeout means dead
        }
        return 1;
    }
    
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
    std::unordered_map<size_t, time_t> heartbeats;
    ThreadPool* threadpool;
    
    Barrier ping_barrier;
    Barrier* train_end_barrier;
    Barrier* terminate_barrier;
    
    Delivery& gDelivery;
};

#endif /* master_h */
