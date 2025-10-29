// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <io/rpc/MessageProcessorBase.h>
#include <io/rpc/MessageUtils.h>

#include <msgpack.hpp>

namespace cloudViewer {
namespace io {
namespace rpc {

/// Message processor implementation which always returns a successful status.
/// This class is meant for testing puproses.
class DummyMessageProcessor : public MessageProcessorBase {
public:
    DummyMessageProcessor() : MessageProcessorBase() {}

    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetMeshData& msg,
            const msgpack::object_handle& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::GetMeshData& msg,
            const msgpack::object_handle& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetCameraData& msg,
            const msgpack::object_handle& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetProperties& msg,
            const msgpack::object_handle& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetActiveCamera& msg,
            const msgpack::object_handle& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetTime& msg,
            const msgpack::object_handle& obj) override {
        return CreateStatusOKMsg();
    }
};

}  // namespace rpc
}  // namespace io
}  // namespace cloudViewer
