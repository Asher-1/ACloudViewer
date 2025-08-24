// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "io/rpc/DummyReceiver.h"

#include <zmq.hpp>

#include "io/rpc/DummyMessageProcessor.h"
#include "io/rpc/Messages.h"

namespace cloudViewer {
namespace io {
namespace rpc {

DummyReceiver::DummyReceiver(const std::string& address, int timeout)
    : ZMQReceiver(address, timeout) {
    SetMessageProcessor(std::make_shared<DummyMessageProcessor>());
}

}  // namespace rpc
}  // namespace io
}  // namespace cloudViewer
