// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ---------------------------------------------------------------------------- ----------------------------------------------------------------------------
// Contains source code from
// https://github.com/mpromonet/webrtc-streamer
//
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
// ----------------------------------------------------------------------------
//
// This is a private header. It shall be hidden from CloudViewer's public API. Do not
// put this in CloudViewer.h.in.

#pragma once

#include <CivetServer.h>
#include <json/json.h>

#include <functional>
#include <map>

namespace cloudViewer {
namespace visualization {
namespace webrtc_server {

class HttpServerRequestHandler : public CivetServer {
public:
    typedef std::function<Json::Value(const struct mg_request_info* req_info,
                                      const Json::Value&)>
            HttpFunction;

    HttpServerRequestHandler(std::map<std::string, HttpFunction>& func,
                             const std::vector<std::string>& options);
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace cloudViewer
