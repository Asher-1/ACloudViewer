// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BagDiscovery.h"

#include <CVLog.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>

namespace fs = std::filesystem;

namespace mcalib {
namespace {

constexpr const char* kTopicGroupSuffixes[] = {
        "Heavy_Topic_Group",
        "Light_Topic_Group",
        "Medium_Topic_Group",
        "Tiny_Topic_Group",
};

bool endsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
                   0;
}

bool isBagFile(const fs::path& path) {
    return fs::is_regular_file(path) && path.extension() == ".bag";
}

void collectBagFilesInDir(const fs::path& dir, std::vector<fs::path>& out) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (isBagFile(entry.path())) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
}

bool hasTopicGroupSubdirs(const fs::path& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return false;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_directory()) continue;
        if (isTopicGroupDirectoryName(entry.path().filename().string())) {
            return true;
        }
    }
    return false;
}

bool hasTopicGroupBagsInDir(const fs::path& dir) {
    std::vector<fs::path> bags;
    collectBagFilesInDir(dir, bags);
    for (const auto& bag : bags) {
        if (!extractTopicGroupSessionKey(bag.stem().string()).empty()) {
            return true;
        }
    }
    return false;
}

fs::path findMergedBagInDir(const fs::path& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return {};
    static const char* kMergedNames[] = {"merge.bag", "sample_aligned.bag"};
    for (const char* name : kMergedNames) {
        const fs::path candidate = dir / name;
        if (fs::is_regular_file(candidate)) {
            return candidate;
        }
    }
    return {};
}

fs::path findTopicGroupBagDir(const fs::path& dir) {
    if (hasTopicGroupBagsInDir(dir)) {
        return dir;
    }
    if (!findMergedBagInDir(dir).empty()) {
        return {};
    }
    static const char* kKnownSubdirs[] = {"orig", "raw_bags", "bags"};
    for (const char* subdir : kKnownSubdirs) {
        const fs::path candidate = dir / subdir;
        if (hasTopicGroupBagsInDir(candidate)) {
            return candidate;
        }
    }
    return {};
}

fs::path nestedLayoutRoot(const fs::path& input) {
    if (fs::is_directory(input)) {
        if (hasTopicGroupSubdirs(input)) return input;
        return {};
    }
    if (!isBagFile(input)) return {};
    const fs::path parent = input.parent_path();
    if (isTopicGroupDirectoryName(parent.filename().string()) &&
        hasTopicGroupSubdirs(parent.parent_path())) {
        return parent.parent_path();
    }
    return {};
}

fs::path flatLayoutRoot(const fs::path& input) {
    if (fs::is_directory(input)) {
        return findTopicGroupBagDir(input);
    }
    if (!isBagFile(input)) return {};
    return findTopicGroupBagDir(input.parent_path());
}

void addBagToSession(std::map<std::string, BagSession>& sessions,
                     const fs::path& bag_path) {
    const std::string stem = bag_path.stem().string();
    if (!isIncludedTopicGroupBag(stem)) {
        return;
    }
    std::string session_key = extractTopicGroupSessionKey(stem);
    if (session_key.empty()) {
        session_key = stem;
    }

    auto& session = sessions[session_key];
    if (session.session_key.empty()) {
        session.session_key = session_key;
    }
    const std::string bag_str = bag_path.string();
    if (std::find(session.bag_paths.begin(), session.bag_paths.end(),
                  bag_str) == session.bag_paths.end()) {
        session.bag_paths.push_back(bag_str);
    }
}

void discoverNestedSessions(const fs::path& root,
                            std::map<std::string, BagSession>& sessions) {
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) continue;
        if (!isTopicGroupDirectoryName(entry.path().filename().string())) {
            continue;
        }
        std::vector<fs::path> bags;
        collectBagFilesInDir(entry.path(), bags);
        for (const auto& bag : bags) {
            addBagToSession(sessions, bag);
        }
    }
}

void discoverFlatSessions(const fs::path& root,
                          std::map<std::string, BagSession>& sessions) {
    std::vector<fs::path> bags;
    collectBagFilesInDir(root, bags);
    for (const auto& bag : bags) {
        addBagToSession(sessions, bag);
    }
}

bool isMergedSingleBagFile(const fs::path& input) {
    return fs::is_regular_file(input) && input.extension() == ".bag" &&
           extractTopicGroupSessionKey(input.stem().string()).empty();
}

BagLayoutType detectLayoutType(const fs::path& input) {
    // Explicit merged/standalone bag file: open directly, never scan siblings.
    if (isMergedSingleBagFile(input)) {
        return BagLayoutType::SingleFile;
    }

    if (nestedLayoutRoot(input).empty() == false) {
        return BagLayoutType::NestedTopicGroup;
    }
    if (flatLayoutRoot(input).empty() == false) {
        return BagLayoutType::FlatTopicGroup;
    }

    if (fs::is_regular_file(input) && input.extension() == ".bag") {
        return BagLayoutType::SingleFile;
    }

    if (fs::is_directory(input)) {
        if (const fs::path merged = findMergedBagInDir(input);
            !merged.empty()) {
            return BagLayoutType::SingleFile;
        }

        std::vector<fs::path> bags;
        collectBagFilesInDir(input, bags);
        if (bags.empty()) {
            return BagLayoutType::Unknown;
        }
        if (bags.size() == 1) {
            return BagLayoutType::SingleFile;
        }
        return BagLayoutType::LegacyMultiBag;
    }

    return BagLayoutType::Unknown;
}

fs::path scanRootForInput(const fs::path& input, BagLayoutType layout) {
    switch (layout) {
        case BagLayoutType::NestedTopicGroup:
            return nestedLayoutRoot(input);
        case BagLayoutType::FlatTopicGroup:
            return flatLayoutRoot(input);
        case BagLayoutType::SingleFile:
            return fs::is_directory(input) ? input : input.parent_path();
        case BagLayoutType::LegacyMultiBag:
            return input;
        default:
            return input;
    }
}

}  // namespace

bool isTopicGroupDirectoryName(const std::string& name) {
    for (const char* suffix : kTopicGroupSuffixes) {
        if (name == suffix) return true;
    }
    return false;
}

std::string extractTopicGroupSessionKey(const std::string& bag_stem) {
    for (const char* suffix : kTopicGroupSuffixes) {
        const std::string marker = std::string(".") + suffix;
        if (endsWith(bag_stem, marker)) {
            return bag_stem.substr(0, bag_stem.size() - marker.size());
        }
    }
    return {};
}

std::string extractTopicGroupName(const std::string& bag_stem) {
    for (const char* suffix : kTopicGroupSuffixes) {
        const std::string marker = std::string(".") + suffix;
        if (endsWith(bag_stem, marker)) {
            return suffix;
        }
    }
    return {};
}

bool isIncludedTopicGroupBag(const std::string& bag_stem) {
    const std::string group = extractTopicGroupName(bag_stem);
    if (group.empty()) {
        return true;
    }
    return group != "Tiny_Topic_Group";
}

std::vector<std::string> filterIncludedSessionBags(
        const std::vector<std::string>& bag_paths) {
    std::vector<std::string> out;
    out.reserve(bag_paths.size());
    for (const auto& path : bag_paths) {
        const fs::path bag_path(path);
        if (isIncludedTopicGroupBag(bag_path.stem().string())) {
            out.push_back(path);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

BagDiscoveryResult discoverBagLayout(const std::string& input_path) {
    BagDiscoveryResult result;
    const fs::path input(input_path);
    if (!fs::exists(input)) {
        result.error_message = "input path does not exist";
        return result;
    }

    result.layout = detectLayoutType(input);
    result.scan_root = scanRootForInput(input, result.layout).string();

    std::map<std::string, BagSession> sessions;
    switch (result.layout) {
        case BagLayoutType::NestedTopicGroup:
            discoverNestedSessions(nestedLayoutRoot(input), sessions);
            break;
        case BagLayoutType::FlatTopicGroup:
            discoverFlatSessions(flatLayoutRoot(input), sessions);
            break;
        case BagLayoutType::SingleFile: {
            fs::path bag_path = input;
            if (fs::is_directory(input)) {
                if (const fs::path merged = findMergedBagInDir(input);
                    !merged.empty()) {
                    bag_path = merged;
                } else {
                    std::vector<fs::path> bags;
                    collectBagFilesInDir(input, bags);
                    if (bags.size() == 1) {
                        bag_path = bags.front();
                    } else {
                        result.error_message = "expected exactly one bag file";
                        result.layout = BagLayoutType::Unknown;
                        return result;
                    }
                }
            }
            BagSession session;
            session.session_key = bag_path.stem().string();
            session.bag_paths.push_back(bag_path.string());
            sessions.emplace(session.session_key, std::move(session));
            break;
        }
        case BagLayoutType::LegacyMultiBag: {
            std::vector<fs::path> bags;
            collectBagFilesInDir(input, bags);
            BagSession session;
            session.session_key = input.filename().string();
            for (const auto& bag : bags) {
                session.bag_paths.push_back(bag.string());
            }
            sessions.emplace(session.session_key, std::move(session));
            break;
        }
        default:
            result.error_message = "unsupported bag input layout";
            return result;
    }

    result.sessions.reserve(sessions.size());
    for (auto& [_, session] : sessions) {
        std::sort(session.bag_paths.begin(), session.bag_paths.end());
        result.sessions.push_back(std::move(session));
    }
    std::sort(result.sessions.begin(), result.sessions.end(),
              [](const BagSession& a, const BagSession& b) {
                  return a.session_key < b.session_key;
              });
    return result;
}

BagResolveResult resolveBagInput(const std::string& input_path,
                                 int session_index,
                                 const std::string& temp_dir) {
    BagResolveResult result;
    result.source_input = input_path;

    BagDiscoveryResult discovery = discoverBagLayout(input_path);
    result.layout = discovery.layout;
    if (discovery.layout == BagLayoutType::Unknown ||
        discovery.sessions.empty()) {
        result.error_message = discovery.error_message.empty()
                                       ? "no bag sessions discovered"
                                       : discovery.error_message;
        return result;
    }

    if (session_index < 0 ||
        session_index >= static_cast<int>(discovery.sessions.size())) {
        result.error_message = "invalid bag session index";
        return result;
    }

    const BagSession& session = discovery.sessions[session_index];
    result.session_key = session.session_key;
    result.source_bags = filterIncludedSessionBags(session.bag_paths);
    if (result.source_bags.empty()) {
        result.error_message =
                "no usable topic-group bags (Heavy/Light/Medium) in session";
        return result;
    }

    result.readable_path = result.source_bags.size() == 1
                                   ? result.source_bags.front()
                                   : std::string{};
    result.ok = true;

    if (result.source_bags.size() == 1) {
        CVLog::Print("[BagDiscovery] open single bag (%s): %s",
                     discovery.layout == BagLayoutType::SingleFile
                             ? "merged/single"
                     : discovery.layout == BagLayoutType::FlatTopicGroup
                             ? "flat topic-group"
                     : discovery.layout == BagLayoutType::NestedTopicGroup
                             ? "nested topic-group"
                             : "legacy",
                     result.readable_path.c_str());
    } else {
        CVLog::Print(
                "[BagDiscovery] open multi-bag session %s (%zu bags: "
                "Heavy/Light/Medium)",
                session.session_key.c_str(), result.source_bags.size());
        for (const auto& bag_path : result.source_bags) {
            CVLog::Print("[BagDiscovery]   %s", bag_path.c_str());
        }
    }
    return result;
}

}  // namespace mcalib
