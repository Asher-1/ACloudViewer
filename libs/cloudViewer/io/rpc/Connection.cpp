// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "io/rpc/Connection.h"

#include <zmq.hpp>

#include "io/rpc/ZMQContext.h"
#include <Logging.h>

using namespace cloudViewer::utility;

namespace {

struct ConnectionDefaults {
    std::string address = "tcp://127.0.0.1:51454";
    int connect_timeout = 5000;
    int timeout = 10000;
} defaults;

}  // namespace

namespace cloudViewer {
namespace io {
namespace rpc {

Connection::Connection()
    : Connection(defaults.address, defaults.connect_timeout, defaults.timeout) {
}

Connection::Connection(const std::string& address,
                       int connect_timeout,
                       int timeout)
    : context_(GetZMQContext()),
      socket_(new zmq::socket_t(*GetZMQContext(), ZMQ_REQ)),
      address_(address),
      connect_timeout_(connect_timeout),
      timeout_(timeout) {
    socket_->set(zmq::sockopt::linger, timeout_);
    socket_->set(zmq::sockopt::connect_timeout, connect_timeout_);
    socket_->set(zmq::sockopt::rcvtimeo, timeout_);
    socket_->set(zmq::sockopt::sndtimeo, timeout_);
    socket_->connect(address_.c_str());
}

Connection::~Connection() { socket_->close(); }

std::shared_ptr<zmq::message_t> Connection::Send(zmq::message_t& send_msg) {
    if (!socket_->send(send_msg, zmq::send_flags::none)) {
        zmq::error_t err;
        if (err.num()) {
            LogInfo("Connection::send() send failed with: {}", err.what());
        }
    }

    std::shared_ptr<zmq::message_t> msg(new zmq::message_t());
    if (socket_->recv(*msg)) {
        LogDebug("Connection::send() received answer with {} bytes",
                 msg->size());
    } else {
        zmq::error_t err;
        if (err.num()) {
            LogInfo("Connection::send() recv failed with: {}", err.what());
        }
    }
    return msg;
}

std::shared_ptr<zmq::message_t> Connection::Send(const void* data,
                                                 size_t size) {
    zmq::message_t send_msg(data, size);
    return Send(send_msg);
}

std::string Connection::DefaultAddress() { return defaults.address; }

}  // namespace rpc
}  // namespace io
}  // namespace cloudViewer
