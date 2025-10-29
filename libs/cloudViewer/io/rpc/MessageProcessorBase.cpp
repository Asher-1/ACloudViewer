// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <io/rpc/MessageProcessorBase.h>
#include <io/rpc/Messages.h>
#include <io/rpc/ZMQContext.h>

#include <zmq.hpp>

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

template <class T>
std::shared_ptr<zmq::message_t> IgnoreMessage(
        const cloudViewer::io::rpc::messages::Request& req,
        const T& msg,
        const msgpack::object_handle& obj) {
    LogInfo("MessageProcessorBase::ProcessMessage: messages with id {} will be "
            "ignored",
            msg.MsgId());
    auto status =
            cloudViewer::io::rpc::messages::Status::ErrorProcessingMessage();
    status.str += ": messages with id " + msg.MsgId() + " are not supported";
    return CreateStatusMessage(status);
}

}  // namespace

namespace cloudViewer {
namespace io {
namespace rpc {

MessageProcessorBase::MessageProcessorBase() {}

MessageProcessorBase::~MessageProcessorBase() {}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetMeshData& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::GetMeshData& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetCameraData& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetProperties& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetActiveCamera& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

std::shared_ptr<zmq::message_t> MessageProcessorBase::ProcessMessage(
        const messages::Request& req,
        const messages::SetTime& msg,
        const msgpack::object_handle& obj) {
    return IgnoreMessage(req, msg, obj);
}

}  // namespace rpc
}  // namespace io
}  // namespace cloudViewer
