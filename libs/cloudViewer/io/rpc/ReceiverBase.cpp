// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 asher-1.github.io
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

#include "io/rpc/ReceiverBase.h"

#include <zmq.hpp>

#include "io/rpc/Messages.h"
#include "io/rpc/ZMQContext.h"

using namespace cloudViewer::utility;

namespace {
std::shared_ptr<zmq::message_t> CreateStatusMessage(
        const cloudViewer::io::rpc::messages::Status& status) {
    msgpack::sbuffer sbuf;
    cloudViewer::io::rpc::messages::Reply reply{status.MsgId()};
    msgpack::pack(sbuf, reply);
    msgpack::pack(sbuf, status);
    std::shared_ptr<zmq::message_t> msg =
            std::make_shared<zmq::message_t>(sbuf.data(), sbuf.size());

    return msg;
}
}  // namespace

namespace cloudViewer {
namespace io {
namespace rpc {

struct ReceiverBase::MsgpackObject {
    explicit MsgpackObject(msgpack::object& obj) : obj_(obj) {}
    msgpack::object& obj_;
};

ReceiverBase::ReceiverBase(const std::string& address, int timeout)
    : address_(address),
      timeout_(timeout),
      keep_running_(false),
      loop_running_(false),
      mainloop_error_code_(0),
      mainloop_exception_("") {}

ReceiverBase::~ReceiverBase() { Stop(); }

void ReceiverBase::Start() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!keep_running_) {
            keep_running_ = true;
            thread_ = std::thread(&ReceiverBase::Mainloop, this);
            // wait for the loop to start running
            while (!loop_running_.load() && !mainloop_error_code_.load()) {
                std::this_thread::yield();
            };

            if (!mainloop_error_code_.load()) {
                LogDebug("ReceiverBase: started");
            }
        } else {
            LogDebug("ReceiverBase: already running");
        }
    }

    if (mainloop_error_code_.load()) {
        LogError(GetLastError().what());
    }
}

void ReceiverBase::Stop() {
    bool keep_running_old;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        keep_running_old = keep_running_;
        if (keep_running_old) {
            keep_running_ = false;
        }
    }
    if (keep_running_old) {
        thread_.join();
        LogDebug("ReceiverBase: stopped");
    } else {
        LogDebug("ReceiverBase: already stopped");
    }
}

std::runtime_error ReceiverBase::GetLastError() {
    const std::lock_guard<std::mutex> lock(mutex_);
    mainloop_error_code_.store(0);
    std::runtime_error result = mainloop_exception_;
    mainloop_exception_ = std::runtime_error("");
    return result;
}

void ReceiverBase::Mainloop() {
    context_ = GetZMQContext();
    socket_ = std::unique_ptr<zmq::socket_t>(
            new zmq::socket_t(*context_, ZMQ_REP));

    socket_->set(zmq::sockopt::linger, 1000);
    socket_->set(zmq::sockopt::rcvtimeo, 1000);
    socket_->set(zmq::sockopt::sndtimeo, timeout_);

    auto limits = msgpack::unpack_limit(0xffffffff,  // array
                                        0xffffffff,  // map
                                        65536,       // str
                                        0xffffffff,  // bin
                                        0xffffffff,  // ext
                                        100          // depth
    );
    try {
        socket_->bind(address_.c_str());
    } catch (const zmq::error_t& err) {
        mainloop_exception_ = std::runtime_error(
                "ReceiverBase::Mainloop: Failed to bind address, " +
                std::string(err.what()));
        mainloop_error_code_.store(1);
        return;
    }

    loop_running_.store(true);
    while (true) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!keep_running_) break;
        }
        try {
            zmq::message_t message;
            if (!socket_->recv(message)) {
                continue;
            }

            const char* buffer = (char*)message.data();
            size_t buffer_size = message.size();

            std::vector<std::shared_ptr<zmq::message_t>> replies;

            size_t offset = 0;
            while (offset < buffer_size) {
                messages::Request req;
                try {
                    auto obj_handle =
                            msgpack::unpack(buffer, buffer_size, offset,
                                            nullptr, nullptr, limits);
                    auto obj = obj_handle.get();
                    req = obj.as<messages::Request>();

                    if (false) {
                    }
#define PROCESS_MESSAGE(MSGTYPE)                                        \
    else if (MSGTYPE::MsgId() == req.msg_id) {                          \
        auto oh = msgpack::unpack(buffer, buffer_size, offset, nullptr, \
                                  nullptr, limits);                     \
        auto obj = oh.get();                                            \
        MSGTYPE msg;                                                    \
        msg = obj.as<MSGTYPE>();                                        \
        auto reply = ProcessMessage(req, msg, MsgpackObject(obj));      \
        if (reply) {                                                    \
            replies.push_back(reply);                                   \
        } else {                                                        \
            replies.push_back(CreateStatusMessage(                      \
                    messages::Status::ErrorProcessingMessage()));       \
        }                                                               \
    }
                    PROCESS_MESSAGE(messages::SetMeshData)
                    PROCESS_MESSAGE(messages::GetMeshData)
                    PROCESS_MESSAGE(messages::SetCameraData)
                    PROCESS_MESSAGE(messages::SetProperties)
                    PROCESS_MESSAGE(messages::SetActiveCamera)
                    PROCESS_MESSAGE(messages::SetTime)
                    else {
                        LogInfo("ReceiverBase::Mainloop: unsupported msg "
                                "id '{}'",
                                req.msg_id);
                        auto status = messages::Status::ErrorUnsupportedMsgId();
                        replies.push_back(CreateStatusMessage(status));
                        break;
                    }
                } catch (std::exception& err) {
                    LogInfo("ReceiverBase::Mainloop:a {}", err.what());
                    auto status = messages::Status::ErrorUnpackingFailed();
                    status.str += std::string(" with ") + err.what();
                    replies.push_back(CreateStatusMessage(status));
                    break;
                }
            }
            if (replies.size() == 1) {
                socket_->send(*replies[0], zmq::send_flags::none);
            } else {
                size_t size = 0;
                for (auto r : replies) {
                    size += r->size();
                }
                zmq::message_t reply(size);
                size_t offset = 0;
                for (auto r : replies) {
                    memcpy((char*)reply.data() + offset, r->data(), r->size());
                    offset += r->size();
                }
                socket_->send(reply, zmq::send_flags::none);
            }
        } catch (const zmq::error_t& err) {
            LogInfo("ReceiverBase::Mainloop: {}", err.what());
        }
    }
    socket_->close();
    loop_running_.store(false);
}

std::shared_ptr<zmq::message_t> ReceiverBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetMeshData& msg,
        const MsgpackObject& obj) {
    LogInfo(
            "ReceiverBase::ProcessMessage: messages with id {} will be "
            "ignored",
            msg.MsgId());
    auto status = messages::Status::ErrorProcessingMessage();
    status.str += ": messages with id " + msg.MsgId() + " are not supported";
    return CreateStatusMessage(status);
}
std::shared_ptr<zmq::message_t> ReceiverBase::ProcessMessage(
        const messages::Request& req,
        const messages::GetMeshData& msg,
        const MsgpackObject& obj) {
    LogInfo(
            "ReceiverBase::ProcessMessage: messages with id {} will be "
            "ignored",
            msg.MsgId());
    auto status = messages::Status::ErrorProcessingMessage();
    status.str += ": messages with id " + msg.MsgId() + " are not supported";
    return CreateStatusMessage(status);
}
std::shared_ptr<zmq::message_t> ReceiverBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetCameraData& msg,
        const MsgpackObject& obj) {
    LogInfo(
            "ReceiverBase::ProcessMessage: messages with id {} will be "
            "ignored",
            msg.MsgId());
    auto status = messages::Status::ErrorProcessingMessage();
    status.str += ": messages with id " + msg.MsgId() + " are not supported";
    return CreateStatusMessage(status);
}
std::shared_ptr<zmq::message_t> ReceiverBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetProperties& msg,
        const MsgpackObject& obj) {
    LogInfo(
            "ReceiverBase::ProcessMessage: messages with id {} will be "
            "ignored",
            msg.MsgId());
    auto status = messages::Status::ErrorProcessingMessage();
    status.str += ": messages with id " + msg.MsgId() + " are not supported";
    return CreateStatusMessage(status);
}
std::shared_ptr<zmq::message_t> ReceiverBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetActiveCamera& msg,
        const MsgpackObject& obj) {
    LogInfo(
            "ReceiverBase::ProcessMessage: messages with id {} will be "
            "ignored",
            msg.MsgId());
    auto status = messages::Status::ErrorProcessingMessage();
    status.str += ": messages with id " + msg.MsgId() + " are not supported";
    return CreateStatusMessage(status);
}
std::shared_ptr<zmq::message_t> ReceiverBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetTime& msg,
        const MsgpackObject& obj) {
    LogInfo(
            "ReceiverBase::ProcessMessage: messages with id {} will be "
            "ignored",
            msg.MsgId());
    auto status = messages::Status::ErrorProcessingMessage();
    status.str += ": messages with id " + msg.MsgId() + " are not supported";
    return CreateStatusMessage(status);
}

}  // namespace rpc
}  // namespace io
}  // namespace cloudViewer
