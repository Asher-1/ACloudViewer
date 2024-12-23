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
