// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <sstream>

#include "io/rpc/ConnectionBase.h"

namespace cloudViewer {
namespace io {
namespace rpc {

/// Implements a connection writing to a buffer
class BufferConnection : public ConnectionBase {
public:
    BufferConnection() {}

    /// Function for sending data wrapped in a zmq message object.
    std::shared_ptr<zmq::message_t> Send(zmq::message_t& send_msg);

    /// Function for sending raw data. Meant for testing purposes
    std::shared_ptr<zmq::message_t> Send(const void* data, size_t size);

    std::stringstream& buffer() { return buffer_; }
    const std::stringstream& buffer() const { return buffer_; }

private:
    std::stringstream buffer_;
};
}  // namespace rpc
}  // namespace io
}  // namespace cloudViewer
