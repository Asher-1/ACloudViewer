// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/io/rpc/MessageProcessorBase.h"

class ccHObject;
namespace cloudViewer {

namespace visualization {

namespace gui {
class Window;
}  // namespace gui

/// MessageProcessor implementation which interfaces with the CloudViewerScene and a
/// Window.
class MessageProcessor : public io::rpc::MessageProcessorBase {
public:
    using OnGeometryFunc =
            std::function<void(std::shared_ptr<ccHObject>,  // geometry
                               const std::string&,          // path
                               int,                         // time
                               const std::string&)>;        // layer
    MessageProcessor(gui::Window* window, OnGeometryFunc on_geometry)
        : MessageProcessorBase(), window_(window), on_geometry_(on_geometry) {}

    std::shared_ptr<zmq::message_t> ProcessMessage(
            const io::rpc::messages::Request& req,
            const io::rpc::messages::SetMeshData& msg,
            const msgpack::object_handle& obj) override;

private:
    gui::Window* window_;
    OnGeometryFunc on_geometry_;

    void SetGeometry(std::shared_ptr<ccHObject> geom,
                     const std::string& path,
                     int time,
                     const std::string& layer);
};

}  // namespace visualization
}  // namespace cloudViewer
