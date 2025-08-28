// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "io/rpc/MessageUtils.h"
#include "io/rpc/ZMQReceiver.h"

namespace cloudViewer {
namespace io {
namespace rpc {

/// Receiver implementation which always returns a successful status.
/// This class is meant for testing puproses.
class DummyReceiver : public ZMQReceiver {
public:
    DummyReceiver(const std::string& address, int timeout);
};

}  // namespace rpc
}  // namespace io
}  // namespace cloudViewer
