//
//  message.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/12/5.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef message_h
#define message_h

#include "../third/zeromq/include/zmq.h"
#include "assert.h"
#include "buffer.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdio>

enum MsgType {
    RESPONSE = 0,
    REQUEST_HANDSHAKE,
    REQUEST_ACK,
    REQUEST_FIN,
    REQUEST_PUSH,
    REQUEST_PULL,
    REQUEST_INFER,
    HEARTBEAT,
    RESERVED,
    UNKNOWN
};

class Package;
class Buffer;
class PackageDescript;

typedef std::function<void(std::shared_ptr<PackageDescript>)> response_callback_t;
typedef std::function<void(std::shared_ptr<PackageDescript>)> sync_barrier_callback_t;

class ZMQ_Message {
public:
    ZMQ_Message() {
        assert(0 == zmq_msg_init(&_zmg));
    }
    ZMQ_Message(char* buf, size_t size) {
        assert(0 == zmq_msg_init_size(&_zmg, size));
        memcpy((void *)buffer(), buf, size);
    }
    
    ZMQ_Message(const ZMQ_Message &) = delete;
    ZMQ_Message(const Buffer& buf) {
        assert(0 == zmq_msg_init_size(&_zmg, buf.size()));
        memcpy((void *)buffer(), buf.buffer(), buf.size());
    }
    
    ~ZMQ_Message() {
        assert(0 == zmq_msg_close(&_zmg));
    }
    
    ZMQ_Message &operator=(const ZMQ_Message &) = delete;
    ZMQ_Message &operator=(ZMQ_Message &&other) {
        if (this != &other) {
            int res = zmq_msg_move(&_zmg, &other.zmg());
            assert(0 == res);
        }
        return *this;
    }
    
    size_t size() {
        return zmq_msg_size(&_zmg);
    }
    
    char *buffer() {
        return (char *)zmq_msg_data(&_zmg);
    }
    
    zmq_msg_t &zmg() {
        return _zmg;
    }
    
private:
    zmq_msg_t _zmg;
};


class PackageDescript {
public:
    // fill by handler
    MsgType msgType;
    size_t epoch_version;
    
    // fill when send
    size_t node_id;
    size_t message_id;
    
    response_callback_t callback;
    sync_barrier_callback_t sync_callback = NULL;
    
    Buffer content;
    
    time_t send_time; // record for timeout monitor
    size_t to_node_id;
    
    ~PackageDescript() {
        
    }
    explicit PackageDescript(MsgType _msgType, size_t _epoch_version = 0)
        : msgType(_msgType), epoch_version(_epoch_version) {
        message_id = 0;
        send_time = 0;
        node_id = to_node_id = -1;
        if (msgType == REQUEST_PUSH) {
            assert(epoch_version > 0);
        }
    }
    PackageDescript &operator=(const PackageDescript &) = delete;
    PackageDescript &operator=(PackageDescript&& other) {
        if (this != &other) {
            msgType = other.msgType;
            epoch_version = other.epoch_version;
            node_id = other.node_id;
            message_id = other.message_id;
            send_time = other.send_time;
            to_node_id = other.to_node_id;
            callback = std::move(other.callback);
            sync_callback = std::move(other.sync_callback);
            other.callback = NULL;
            other.sync_callback = NULL;
            content = std::move(other.content);
        }
        return *this;
    }
    PackageDescript(const PackageDescript& other) { // copy only by constructor
        msgType = other.msgType;
        epoch_version = other.epoch_version;
        node_id = other.node_id;
        message_id = other.message_id;
        send_time = other.send_time;
        to_node_id = other.to_node_id;
        callback = other.callback;
        sync_callback = other.sync_callback;
        content = Buffer(other.content.buffer(), other.content.size());
    }
    PackageDescript(PackageDescript&& other) {
        msgType = other.msgType;
        epoch_version = other.epoch_version;
        node_id = other.node_id;
        message_id = other.message_id;
        send_time = other.send_time;
        to_node_id = other.to_node_id;
        callback = std::move(other.callback);
        sync_callback = std::move(other.sync_callback);
        other.callback = NULL;
        other.sync_callback = NULL;
        content = std::move(other.content);
    }
    
    bool operator==(const PackageDescript& other) const {
        if (message_id == other.message_id) {
            return true;
        }
        return false;
    }
};

const size_t _Head_size = sizeof(MsgType) + 3 * sizeof(size_t);

class Package {
public:
    Package() {
    }
    Package(const PackageDescript& pDesc) {
        head = ZMQ_Message((char *)&pDesc, _Head_size);
        content = ZMQ_Message(pDesc.content);
    }
    
    void Descript(std::shared_ptr<PackageDescript>& pDesc) {
        pDesc = std::make_shared<PackageDescript>(PackageDescript(UNKNOWN));
        assert(pDesc);
        assert(head.size() == _Head_size);
        memcpy(pDesc.get(), head.buffer(), _Head_size);
        pDesc->content = Buffer(content.buffer(), content.size());
    }
    
    Package &operator=(const Package &) = delete;
    Package(const Package &) = delete;
    Package(Package &&other) {
        head = std::move(other.head);
        content = std::move(other.content);
    }
    
    ZMQ_Message head;
    ZMQ_Message content;
};

#endif /* message_h */
