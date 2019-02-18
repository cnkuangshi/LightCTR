//
//  network.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef network_h
#define network_h

#include "../third/zeromq/include/zmq.h"
#include "system.h"
#include "thread_pool.h"
#include "lock.h"
#include "barrier.h"
#include "message.h"
#include "message_queue.h"
#include "assert.h"

#include <sstream>
#include <stdio.h>
#include <map>

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

#define BEGIN_ID_OF_WORKER 10000
#define BEGIN_ID_OF_PS 1

const std::string __global_Master_IP_Port = "tcp://" +
                                            std::string(getEnv("LightCTR_MASTER_ADDR",
                                                   "127.0.0.1:17832"));

typedef std::function<void(std::shared_ptr<PackageDescript>, PackageDescript&)> request_handler_t;

std::string get_local_ip() {
    struct ifaddrs *ifAddrStruct = NULL;
    struct ifaddrs *ifa = NULL;
    void *tmpAddrPtr = NULL;
    std::string local_ip;
    
    getifaddrs(&ifAddrStruct);
    for (ifa = ifAddrStruct; ifa; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) { // IPv4 Address
            tmpAddrPtr = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            char address[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmpAddrPtr, address, INET_ADDRSTRLEN);
            if (strcmp(address, "127.0.0.1") != 0 &&
                strcmp(address, "0.0.0.0") != 0) {
                local_ip = address;
            }
        }
    }
    if (ifAddrStruct)
        freeifaddrs(ifAddrStruct);
    return local_ip;
}

struct Addr {
    uint16_t addr[4] = {0};
    uint16_t port = 0;
    
    Addr() {
    }
    Addr(const char *_addr) {
        const char *begin = _addr + 6;
        char *end;
        for (int i = 0; i < 4; i++) {
            addr[i] = (uint16_t)std::strtoul(begin, &end, 10);
            begin = end + 1;
        }
        port = (uint16_t)std::strtoul(begin, &end, 10);
    }
    Addr(Buffer &msg) {
        assert(*msg.cursor() == 't');
        msg.cursor_preceed(6); // skip tcp://
        char *end;
        for (int i = 0; i < 4; i++) {
            addr[i] = (uint16_t)std::strtoul(msg.cursor(), &end, 10);
            const size_t offset = end - msg.cursor() + 1;
            msg.cursor_preceed(offset);
        }
        port = (uint16_t)std::strtoul(msg.cursor(), &end, 10);
        const size_t offset = std::min(end - msg.cursor(), msg.end() - msg.cursor());
        msg.cursor_preceed(offset);
    }
    
    Addr(const Addr &) = delete;
    Addr(Addr &&other) {
        for (int i = 0; i < 4; i++) {
            addr[i] = other.addr[i];
        }
        port = other.port;
    }
    
    Addr &operator=(const Addr &) = delete;
    Addr &operator=(const Addr &&other) {
        for (int i = 0; i < 4; i++) {
            addr[i] = other.addr[i];
        }
        port = other.port;
        return *this;
    }
    
    std::string toString() const {
        std::stringstream ss;
        ss << "tcp://";
        ss << addr[0];
        for (int i = 1; i < 4; i++) {
            ss << "." << addr[i];
        }
        ss << ":" << port;
        return ss.str();
    }
};

class Delivery {
public:
    static Delivery& Instance() { // singleton
        static std::once_flag once;
        static Delivery delivery;
        std::call_once(once, [] {
            delivery.start_loop();
        });
        return delivery;
    }
    
    void regist_router(size_t node_id, Addr&& addr) {
        void* socket = zmq_socket(zmq_ctx, ZMQ_PUSH);
        assert(socket);
        int res = 0, retry_conn = 3;
        while (retry_conn--) { // retry to connect
            res = zmq_connect(socket, addr.toString().c_str());
            if (0 == res) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        assert(0 == res);
        
        std::unique_lock<SpinLock> glock(router_lock);
        if (0 != router_socket.count(node_id)) {
            printf("[Router] %zu is Re-registering\n", node_id);
            router_addr[node_id] = std::forward<Addr>(addr);
            router_socket[node_id] = socket;
        } else {
            router_addr.emplace(node_id, std::forward<Addr>(addr));
            router_socket.emplace(node_id, socket);
        }
        
        printf("[Router] Add node_id = %zu addr = %s\n",
               node_id, addr.toString().c_str());
    }
    
    const Addr& get_router(size_t node_id) {
        if (node_id == cur_node_id) {
            return listen_addr;
        }
        std::unique_lock<SpinLock> glock(router_lock);
        assert(router_addr.count(node_id) != 0);
        return router_addr[node_id];
    }
    
    bool delete_router(size_t node_id) {
        std::unique_lock<SpinLock> glock(router_lock);
        if(0 == router_socket.count(node_id)) {
            return false;
        }
        assert(0 == zmq_close(router_socket[node_id]));
        router_addr.erase(node_id);
        router_socket.erase(node_id);
        printf("[Router] Delete node_id = %zu\n", node_id);
        return true;
    }
    
    void regist_handler(MsgType type, request_handler_t&& handler) {
        std::unique_lock<SpinLock> lock(handlerMap_lock);
        if (handlerMap.count(type) == 0) {
            handlerMap.emplace(type, std::forward<request_handler_t>(handler));
        } else {
            handlerMap[type] = std::forward<request_handler_t>(handler);
        }
    }
    
    void send_async(PackageDescript& pDesc, size_t to_id) {
        pDesc.node_id = cur_node_id;
        pDesc.to_node_id = to_id;
        pDesc.send_time = time(NULL);
        
        if (pDesc.msgType != RESPONSE) {
            // new msg_id will skip RESPONSE
            pDesc.message_id = msg_seq++;
            
            if (pDesc.msgType != HEARTBEAT) {
                // new resend_queue will skip HEARTBEAT and RESPONSE
                // Never resend RESPONSE
                if(time(NULL) - pDesc.send_time <= 20) {
                    sending_queue.push(pDesc); // save Duplicated package into queue
#ifdef DEBUG
                    printf("[QUEUE] save package msg_id = %zu msg_remain = %zu\n",
                           pDesc.message_id, sending_queue.size());
#endif
                }
            }
        }
        if (pDesc.sync_callback) {
            callbackMap.emplace(pDesc.message_id, std::move(pDesc.sync_callback));
        } else if(pDesc.callback) {
            std::unique_lock<SpinLock> lock(callbackMap_lock);
            assert(callbackMap.count(pDesc.message_id) == 0);
            callbackMap.emplace(pDesc.message_id, std::move(pDesc.callback));
        }
#ifdef DEBUG
        printf("[Network] Sending to node_id = %zu msg_id = %zu msgType = %d\n",
               to_id, pDesc.message_id, pDesc.msgType);
#endif
        Package snd_package(pDesc);
        assert(snd_package.head.size() > 0);
        
        const size_t pkg_size = snd_package.head.size();
        {
            // turn package sequence of channel into serial communication
            std::unique_lock<SpinLock> glock(router_lock);
            auto it = router_socket.find(to_id);
            if (it == router_socket.end()) {
                return; // double check whether node serving
            }
            int res = zmq_msg_send(&snd_package.head.zmg(), it->second, ZMQ_SNDMORE);
            assert(res == pkg_size);
            res = zmq_msg_send(&snd_package.content.zmg(), it->second, 0);
            assert(res >= 0);
        }
    }
    
    bool send_sync(PackageDescript& pDesc, size_t to_id, time_t RTT_timeout_ms) {
        assert(pDesc.msgType != RESPONSE);
        Barrier barrier;
        pDesc.sync_callback = [&pDesc, &barrier](std::shared_ptr<PackageDescript> ptr) {
            if (pDesc.callback)
                pDesc.callback(ptr);
            barrier.unblock();
        };
        send_async(pDesc, to_id);
        return barrier.block(RTT_timeout_ms, NULL);
    }
    
    void start_loop() {
#if (defined PS) || (defined WORKER) || (defined WORKER_RING)
        assert(0 == listen_bind());
#else
        int res = zmq_bind(listen_socket, __global_Master_IP_Port.c_str());
        assert(res == 0);
        listen_addr = Addr(__global_Master_IP_Port.c_str());
        printf("[Network] Listening %s\n", __global_Master_IP_Port.c_str());
#endif
        for (int i = 0; i < concurrent_cnt; i++) {
            // io_pool only handle zmq io event
            io_pool->addTask(std::bind(&Delivery::event_loop, this));
        }
    }
    
    void shutdown() {
        if (!serving) {
            return;
        }
        serving = false;
        
        sending_queue.push(PackageDescript(BREAKER));
        RTT_timeout_monitor->join();
        puts("[Network] stop RTT timeout monitor");
        handle_pool->wait();
        puts("[Network] stop handlers");
        callback_pool->wait();
        puts("[Network] stop message callback");
        
        // break event loop
        puts("[Network] prepare to break event loop");
        void* socket = zmq_socket(zmq_ctx, ZMQ_PUSH);
        assert(socket);
        assert(0 == zmq_connect(socket, listen_addr.toString().c_str()));
        for (int i = 0; i < concurrent_cnt; i++) {
            assert(0 == zmq_msg_send(&ZMQ_Message().zmg(), socket, 0));
        }
        
        io_pool->wait();
        puts("[Network] stop IO Eventloop");
        
        puts("[Network] shutdown complete");
    }
    
    inline size_t node_id() const {
        return cur_node_id;
    }
    void set_node_id(size_t node_id) {
        cur_node_id = node_id;
    }
    inline const Addr& local_addr() const {
        return listen_addr;
    }

private:
    Delivery() {
        zmq_ctx = zmq_ctx_new();
        init();
    }
    explicit Delivery(void *_zmq_ctx) {
        zmq_ctx = _zmq_ctx;
        init();
    }
    
    ~Delivery() {
        shutdown();
        delete handle_pool;
        handle_pool = NULL;
        delete callback_pool;
        callback_pool = NULL;
        delete RTT_timeout_monitor;
        RTT_timeout_monitor = NULL;
        
        if (listen_socket) {
            assert(0 == zmq_close(listen_socket));
            listen_socket = NULL;
        }
        for (auto &route : router_socket) {
            assert(0 == zmq_close(route.second));
        }
        assert(0 == zmq_ctx_destroy(zmq_ctx));
    }
    Delivery(const Delivery&) = delete;
    Delivery(Delivery&&) = delete;
    Delivery &operator=(const Delivery &) = delete;
    Delivery &operator=(Delivery &&) = delete;
    
    void init() {
        // first regist master addr and regist cur node to master
        // master addr should be config
#if (defined PS) || (defined WORKER) || (defined WORKER_RING)
        Addr master_addr(__global_Master_IP_Port.c_str());
        regist_router(0, std::move(master_addr)); // reserve 0 for master
#endif
        
        cur_node_id = 0; // init with 1 for ps and BEGIN_ID_OF_WORKER for worker
        
        listen_socket = zmq_socket(zmq_ctx, ZMQ_PULL);
        assert(listen_socket);
        concurrent_cnt = std::thread::hardware_concurrency();
        assert(concurrent_cnt > 1);
        io_pool = new ThreadPool(concurrent_cnt);
        handle_pool = new ThreadPool(concurrent_cnt / 2);
        callback_pool = new ThreadPool(1); // callback call serialize
        // TODO whether improve degree of parallelism
        
        // monitor timeout to get response and retry to request
        RTT_timeout_monitor = new std::thread(&Delivery::timeoutResender, this);
    }
    
    int listen_bind() {
        while (1) {
            std::string ip = get_local_ip();
            std::stringstream addr;
            addr << "tcp://";
            addr << ip << ":";
            addr << 1024 + rand() % (65536 - 1024);
            std::string addr_str = addr.str();
            int res = zmq_bind(listen_socket, addr_str.c_str());
            if (res == 0) {
                listen_addr = Addr(addr_str.c_str());
                printf("[Network] Listening %s\n", addr_str.c_str());
                return 0;
            }
            assert(errno == EADDRINUSE); // assert other error
        }
        return -1;
    }
    
    void event_loop() {
        Package recv_package;
        while(serving) {
            {
                // long wait use mutex
                std::unique_lock<std::mutex> lock(listen_mutex);
                int res = zmq_msg_recv(&recv_package.head.zmg(), listen_socket, 0);
                if (res < 0 && !serving) {
                    break;
                }
                assert(res >= 0);
                if (res == 0) { // shutdown message
                    break;
                }
                assert(zmq_msg_more(&recv_package.head.zmg()));
                res = zmq_msg_recv(&recv_package.content.zmg(), listen_socket, 0);
                assert(res >= 0);
            }
            
            std::shared_ptr<PackageDescript> ptr;
            recv_package.Descript(ptr);
            if (ptr->msgType == RESPONSE) { // handle PS's response
                handle_response(ptr);
            } else { // handle workers' PULL & PUSH request
                handle_request(ptr);
            }
        }
    }
    
    void handle_request(std::shared_ptr<PackageDescript> request) {
#ifdef DEBUG
        printf("[REQUEST] Receiving from node_id = %zu msg_id = %zu msgType = %d\n",
               request->node_id, request->message_id, request->msgType);
#endif
        request_handler_t handler; // get handler
        if (request->msgType != HEARTBEAT) {
            std::unique_lock<SpinLock> lock(handlerMap_lock);
            auto it = handlerMap.find(request->msgType);
            if (it == handlerMap.end()) {
#ifdef DEBUG
                puts("[WARNING][Network] Recv Unacceptable msg, Skip and No response");
#endif
                return;
            }
            handler = it->second;
        }
        handle_pool->addTask([this, handler, request]() {
            // copy handler
            PackageDescript resp_desc(RESPONSE);
            resp_desc.message_id = request->message_id; // keep resp msgid
            if (handler) {
                handler(request, resp_desc); // fill response msg
            }
            const size_t to_id = request->node_id;
            resp_desc.to_node_id = to_id;
#ifdef DEBUG
            printf("[RESPONSE]");
#endif
            send_async(resp_desc, to_id); // send response
        });
    }
    
    void handle_response(std::shared_ptr<PackageDescript> response) {
#ifdef DEBUG
        printf("[RESPONSE] msg_id = %zu msgType = %d\n",
               response->message_id, response->msgType);
#endif
        int res = sending_queue.erase(*response);
        if (res == 1) {
#ifdef DEBUG
            printf("[QUEUE] ACK req msg_id = %zu msg_remain = %zu\n",
                   response->message_id, sending_queue.size());
#endif
        }
        response_callback_t callback;
        {
            std::unique_lock<SpinLock> lock(callbackMap_lock);
            auto it = callbackMap.find(response->message_id);
            if (it != callbackMap.end()) {
                callback = std::move(it->second);
                callbackMap.erase(it); // handler used once
            }
        }
        callback_pool->addTask([response, callback]() {
            // copy callback and pointer
            if (callback) {
                callback(response); // callback don't need thread-safe
            }
        });
    }
    
    void timeoutResender() {
        const time_t timeout = 2; // seconds
        const size_t max_retry_times = 5;
        while(serving) {
            const PackageDescript& pkg = sending_queue.front();
            if (pkg.msgType == BREAKER) {
                return;
            }
            const size_t pkg_msgid = pkg.message_id;
            const size_t pkg_to_nodeid = pkg.to_node_id;
            assert(pkg.send_time > 0);
            
            time_t cur_time = time(NULL);
            if (pkg.send_time + max_retry_times * timeout >= cur_time) {
                sending_queue.pop();
                continue;
            } else if (pkg.send_time + timeout >= cur_time) {
                const time_t interval = pkg.send_time + timeout - cur_time;
                assert(interval >= 0);
                std::this_thread::sleep_for(std::chrono::seconds(interval));
            }
            if (!serving)
                break;
            PackageDescript resend_pkg(UNKNOWN);
            resend_pkg.message_id = pkg_msgid;
            resend_pkg.to_node_id = pkg_to_nodeid;
            if (sending_queue.pop_if(resend_pkg, &resend_pkg)) {
                // detect timeout, do resend
#ifdef DEBUG
                printf("[Re-Send]");
#endif
                send_async(resend_pkg, resend_pkg.to_node_id);
            }
        }
    }
    
    std::atomic<bool> serving{true};
    ThreadPool *io_pool, *handle_pool, *callback_pool;
    
    void *zmq_ctx;
    
    std::mutex listen_mutex;
    Addr listen_addr;
    void *listen_socket;
    
    size_t cur_node_id;
    
    size_t concurrent_cnt;
    std::atomic<size_t> msg_seq{1};
    
    SpinLock handlerMap_lock;
    std::map<MsgType, request_handler_t> handlerMap;
    SpinLock callbackMap_lock;
    std::map<size_t, response_callback_t> callbackMap;
    
    std::map<size_t, void *> router_socket;
    std::map<size_t, Addr> router_addr;
    SpinLock router_lock;
    
    std::thread* RTT_timeout_monitor;
    MessageQueue<PackageDescript> sending_queue;
};

#endif /* network_h */
