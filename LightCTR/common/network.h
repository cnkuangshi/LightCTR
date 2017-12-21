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

const std::string __global_Master_IP_Port = getEnv("LightCTR_MASTER_ADDR",
                                                   "tcp://127.0.0.1:17832");

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
    Delivery() {
        zmq_ctx = zmq_ctx_new();
        init();
    }
    explicit Delivery(void *_zmq_ctx) {
        zmq_ctx = _zmq_ctx;
        init();
    }
    
    ~Delivery() {
        assert(!serving); // should call shutdown first
        handle_pool->join();
        callback_pool->join();
        
        if (listen_socket) {
            assert(0 == zmq_close(listen_socket));
            listen_socket = NULL;
        }
        for (auto &route : router_socket) {
            assert(0 == zmq_close(route.second));
        }
        assert(0 == zmq_ctx_destroy(zmq_ctx));
    }
    Delivery &operator=(const Delivery &) = delete;
    
    static Delivery&& Instance() { // singleton
        static std::once_flag once;
        static Delivery delivery;
        std::call_once(once, [] {
            delivery.start_loop();
        });
        return std::move(delivery);
    }
    
    void regist_router(size_t node_id, Addr& addr) {
        if (0 != router_socket.count(node_id)) {
            printf("[Router] %zu is Re-registering\n", node_id);
        }
        
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
        
        router_addr.emplace(node_id, std::move(addr));
        router_lock.emplace(node_id, std::shared_ptr<SpinLock>(new SpinLock()));
        router_socket.emplace(node_id, socket);
        
        printf("[Router] Add node_id = %zu addr = %s\n",
               node_id, addr.toString().c_str());
    }
    
    const Addr&& get_router(size_t node_id) {
        if (node_id == cur_node_id) {
            return std::move(listen_addr);
        }
        assert(router_addr.count(node_id) != 0);
        return std::move(router_addr[node_id]);
    }
    
    void delete_router(size_t node_id) {
        if(0 == router_socket.count(node_id)) {
            return;
        }
        assert(0 == zmq_close(router_socket[node_id]));
        router_addr.erase(node_id);
//        router_lock.erase(node_id);
        router_socket.erase(node_id);
        printf("[Router] Delete node_id = %zu\n", node_id);
    }
    
    void regist_handler(MsgType type, request_handler_t &&handler) {
        std::unique_lock<SpinLock> lock(handlerMap_lock);
        assert(handlerMap.count(type) == 0);
        handlerMap.emplace(type, std::move(handler));
    }
    
    void send_sync(PackageDescript&& pDesc, size_t to_id) {
        pDesc.node_id = cur_node_id;
        pDesc.to_node_id = to_id;
        pDesc.send_time = time(NULL);
        
        if (pDesc.msgType != RESPONSE) {
            pDesc.message_id = msg_seq++;
            
            if (sending_queue && pDesc.msgType != HEARTBEAT) {
                // new msg_id and resend_queue will skip RESPONSE and HEARTBEAT
                sending_queue->push(pDesc); // save Duplicated package into queue
                printf("[QUEUE] save package msg_id = %zu msg_remain = %zu\n",
                       pDesc.message_id, sending_queue->size());
            }
        }
        
        if(pDesc.callback) {
            std::unique_lock<SpinLock> lock(callbackMap_lock);
            assert(callbackMap.count(pDesc.message_id) == 0);
            callbackMap.emplace(pDesc.message_id, std::move(pDesc.callback));
        }
        
        printf("[Network] Sending to node_id = %zu msg_id = %zu msgType = %d\n",
               to_id, pDesc.message_id, pDesc.msgType);
        
        Package snd_package(pDesc);
        assert(snd_package.head.size() > 0);
        
        const size_t pkg_size = snd_package.head.size();
        {
            assert(1 == router_lock.count(to_id));
            // turn package sequence of channel into serial communication
            std::unique_lock<SpinLock> glock(*router_lock[to_id]);
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
    
    void start_loop() {
#if (defined PS) || (defined WORKER)
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
        
        // break event loop
        void* socket = zmq_socket(zmq_ctx, ZMQ_PUSH);
        assert(socket);
        assert(0 == zmq_connect(socket, listen_addr.toString().c_str()));
        for (int i = 0; i < concurrent_cnt; i++) {
            assert(0 == zmq_msg_send(&ZMQ_Message().zmg(), socket, 0));
        }
        
        io_pool->join();
        puts("[Network] shutdown");
    }
    
    inline size_t node_id() const {
        return cur_node_id;
    }
    void set_node_id(size_t node_id) {
        cur_node_id = node_id;
    }
    inline const Addr& local_addr() {
        return listen_addr;
    }

private:
    void init() {
        // first regist master addr and regist cur node to master
        // master addr should be config
#if (defined PS) || (defined WORKER)
        Addr master_addr(__global_Master_IP_Port.c_str());
        regist_router(0, master_addr); // reserve 0 for master
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
        sending_queue = new MessageQueue<PackageDescript>();
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
        printf("[REQUEST] recv msg_id = %zu msgType = %d\n",
               request->message_id, request->msgType);
        
        request_handler_t handler; // get handler of PS
        if (request->msgType != HEARTBEAT) {
            std::unique_lock<SpinLock> lock(handlerMap_lock);
            auto it = handlerMap.find(request->msgType);
            if (it == handlerMap.end()) {
                puts("[Network] Recv unacceptable msg, Skip");
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
            printf("[RESPONSE]");
            send_sync(std::move(resp_desc), to_id); // send response
        });
    }
    
    void handle_response(std::shared_ptr<PackageDescript> response) {
        
        printf("[RESPONSE] msg_id = %zu msgType = %d\n",
               response->message_id, response->msgType);
        
        if (sending_queue) {
            int res = sending_queue->erase(*response);
            if (res == 1) {
                printf("[QUEUE] ACK req msg_id = %zu msg_remain = %zu\n",
                       response->message_id, sending_queue->size());
            }
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
        callback_pool->addTask([this, response, callback]() {
            // copy callback and pointer
            if (callback) {
                callback(response); // callback don't need thread-safe
            }
        });
    }
    
    void timeoutResender() {
        const time_t timeout = 10;
        while(serving) {
            if (!sending_queue) {
                return;
            }
            const PackageDescript& pkg = sending_queue->front();
            const size_t pkg_msgid = pkg.message_id;
            const size_t pkg_to_nodeid = pkg.to_node_id;
            assert(pkg.send_time > 0);
            
            time_t cur_time = time(NULL);
            if (pkg.send_time + timeout > cur_time) {
                const time_t interval = pkg.send_time + timeout - cur_time;
                assert(interval > 0);
                std::this_thread::sleep_for(std::chrono::seconds(interval));
            }
            PackageDescript resend_pkg(RESERVED);
            resend_pkg.message_id = pkg_msgid;
            resend_pkg.to_node_id = pkg_to_nodeid;
            if (sending_queue->pop_if(resend_pkg, &resend_pkg)) {
                // detect timeout, do resend
                printf("[Re-Send]");
                send_sync(std::move(resend_pkg), resend_pkg.to_node_id);
            }
        }
    }
    
    bool serving = true;
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
    std::map<size_t, std::shared_ptr<SpinLock> > router_lock;
    
    std::thread* RTT_timeout_monitor;
    MessageQueue<PackageDescript>* sending_queue;
};

#endif /* network_h */
