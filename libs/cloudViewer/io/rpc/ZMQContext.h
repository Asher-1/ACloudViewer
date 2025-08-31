// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

namespace zmq {
class context_t;
}

namespace cloudViewer {
namespace io {
namespace rpc {

/// Returns the zeromq context for this process.
std::shared_ptr<zmq::context_t> GetZMQContext();

/// Destroys the zeromq context for this process. On windows this needs to be
/// called manually for a clean shutdown of the process.
void DestroyZMQContext();

}  // namespace rpc
}  // namespace io
}  // namespace cloudViewer
