// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>
#include <string>
#include <vector>

#include "RosBagReader.h"

namespace mcalib {

/// Minimal ROS bag v2 writer (uncompressed chunks) for merge/export workflows.
class RosBagWriter {
public:
    RosBagWriter();
    ~RosBagWriter();

    bool open(const std::string& path);
    bool isOpen() const { return is_open_; }

    uint32_t addConnection(const BagConnection& conn);
    bool writeMessage(uint32_t conn_id,
                      uint64_t timestamp_ns,
                      const std::string& data);
    bool close();

private:
    struct PendingMessage {
        uint32_t conn_id = 0;
        uint64_t timestamp_ns = 0;
        std::string data;
    };

    std::string path_;
    bool is_open_ = false;
    uint32_t next_conn_id_ = 0;
    std::map<std::string, uint32_t> topic_to_conn_;
    std::map<uint32_t, BagConnection> connections_;
    std::vector<PendingMessage> messages_;
};

bool mergeRosBags(const std::vector<std::string>& input_bags,
                  const std::string& output_bag);

/// Copy messages in [start_ns, end_ns] without re-encoding payload bytes.
bool filterRosBagByTime(const std::string& input_bag,
                        const std::string& output_bag,
                        uint64_t start_ns,
                        uint64_t end_ns);

}  // namespace mcalib
