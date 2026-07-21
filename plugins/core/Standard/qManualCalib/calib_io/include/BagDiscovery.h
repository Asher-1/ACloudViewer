// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

namespace mcalib {

enum class BagLayoutType {
    Unknown = 0,
    SingleFile,
    FlatTopicGroup,
    NestedTopicGroup,
    LegacyMultiBag,
};

struct BagSession {
    std::string session_key;
    std::vector<std::string> bag_paths;
};

struct BagDiscoveryResult {
    BagLayoutType layout = BagLayoutType::Unknown;
    std::vector<BagSession> sessions;
    std::string scan_root;
    std::string error_message;
};

struct BagResolveResult {
    bool ok = false;
    std::string readable_path;
    std::string source_input;
    BagLayoutType layout = BagLayoutType::Unknown;
    std::string session_key;
    std::vector<std::string> source_bags;
    std::string merged_temp_path;
    std::string error_message;
};

/// Scan a bag file or directory and group bags into recording sessions.
BagDiscoveryResult discoverBagLayout(const std::string& input_path);

/// Resolve input to a single readable bag path (merge when needed).
/// @param session_index Index into discovery.sessions (default 0).
/// @param temp_dir Optional directory for merged bag; uses system temp if
/// empty.
BagResolveResult resolveBagInput(const std::string& input_path,
                                 int session_index = 0,
                                 const std::string& temp_dir = {});

/// Extract session key from a topic-group bag filename stem.
/// Returns empty string when the name does not match the topic-group pattern.
std::string extractTopicGroupSessionKey(const std::string& bag_stem);

/// Extract topic-group suffix (Heavy_Topic_Group, etc.) from bag stem.
std::string extractTopicGroupName(const std::string& bag_stem);

/// Heavy/Light/Medium are required; Tiny is ignored.
bool isIncludedTopicGroupBag(const std::string& bag_stem);

/// Known topic-group subdirectory names (Heavy_Topic_Group, etc.).
bool isTopicGroupDirectoryName(const std::string& name);

/// Filter session bags to Heavy/Light/Medium only.
std::vector<std::string> filterIncludedSessionBags(
        const std::vector<std::string>& bag_paths);

}  // namespace mcalib
