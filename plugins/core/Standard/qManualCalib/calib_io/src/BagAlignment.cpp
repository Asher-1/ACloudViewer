// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BagAlignment.h"

#include <CVLog.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <future>
#include <set>
#include <vector>

#include "CloudDecombine.h"
#include "ProtoDecoder.h"
#include "RosBagReader.h"
#include "RosBagWriter.h"
#include "VideoCodecDecoder.h"

namespace mcalib {

namespace {

constexpr const char* kCarStateTopic = "/canbus/car_state";

int64_t timeDiffNs(int64_t a, int64_t b) { return a > b ? a - b : b - a; }

constexpr int64_t kBagProtoMismatchNs = 60LL * 1000000000LL;

int64_t imageSyncStampNs(const BagMessage& msg) {
    const int64_t bag_ns = static_cast<int64_t>(msg.timestamp_ns);
    double ts_sec = 0;
    if (!ProtoDecoder::extractCompressedImageTimestampFromBag(msg.data,
                                                              ts_sec) ||
        ts_sec <= 0) {
        return bag_ns;
    }
    const int64_t proto_ns = static_cast<int64_t>(ts_sec * 1e9);
    if (bag_ns <= 0) return proto_ns;
    if (timeDiffNs(proto_ns, bag_ns) > kBagProtoMismatchNs) return bag_ns;
    return proto_ns;
}

int64_t cloudSyncStampNs(const BagMessage& msg) {
    const int64_t bag_ns = static_cast<int64_t>(msg.timestamp_ns);
    double ts_sec = 0;
    if (!ProtoDecoder::extractPointCloudTimestampFromBag(msg.data, ts_sec) ||
        ts_sec <= 0) {
        return bag_ns;
    }
    const int64_t proto_ns = static_cast<int64_t>(ts_sec * 1e9);
    if (bag_ns <= 0) return proto_ns;
    if (timeDiffNs(proto_ns, bag_ns) > kBagProtoMismatchNs) return bag_ns;
    return proto_ns;
}

bool messageNeedsVideoDecode(const BagMessage& msg);

// Forward declarations for helpers defined further down in this file.
std::string pickReferenceImageTopic(const std::vector<std::string>& topics);
bool decodeAlignedImageMessages(
        RosBagReader& reader,
        const std::vector<BagMessage>& msgs,
        std::map<std::string, cv::Mat>& images_by_topic,
        std::map<std::string, int64_t>& stamps_ns_by_topic);
bool syncVehicleStateNearStamp(RosBagReader& reader,
                               int64_t ref_bag_stamp_ns,
                               int64_t ref_proto_stamp_ns,
                               VehicleStateData& vehicle_state);

constexpr int64_t kImageSearchRangeNs =
        1000000000;  // 1s (codetree manual_sensor_calib)

// codetree/repo/calibration/.../manual_sensor_calib.cpp:get_aligned_images
void computeImageSearchWindow(RosBagReader& reader,
                              double percent,
                              uint64_t& start_ns,
                              uint64_t& end_ns) {
    const uint64_t bag_begin = reader.getBeginTime();
    const uint64_t bag_end = reader.getEndTime();
    const uint64_t bag_range = reader.getDuration();

    const uint64_t target_ns =
            bag_begin + static_cast<uint64_t>(bag_range * percent);

    int64_t half_window = kImageSearchRangeNs / 2;
    if (bag_range > 0 && bag_range < kImageSearchRangeNs) {
        // Remapped short bags (e.g. 3-frame ~0.6s slices): a 1s window covers
        // the entire bag and always picks the first sync group.
        half_window = static_cast<int64_t>(bag_range / 6);
        half_window = std::max<int64_t>(half_window, 20000000LL);  // >= 20ms
    }

    int64_t start = static_cast<int64_t>(target_ns) - half_window;
    int64_t end = static_cast<int64_t>(target_ns) + half_window;

    start_ns = static_cast<uint64_t>(
            std::max<int64_t>(start, static_cast<int64_t>(bag_begin)));
    end_ns = static_cast<uint64_t>(
            std::min<int64_t>(end, static_cast<int64_t>(bag_end)));
    if (end_ns <= start_ns) {
        end_ns = std::min(bag_end, start_ns + 1);
    }
}

void computeSearchWindow(RosBagReader& reader,
                         double percent,
                         uint64_t& start_ns,
                         uint64_t& end_ns) {
    computeImageSearchWindow(reader, percent, start_ns, end_ns);
}

std::vector<std::string> selectCloudTopics(
        RosBagReader& reader, const std::vector<std::string>& candidates) {
    const auto available = reader.getTopics();
    std::set<std::string> topic_set(available.begin(), available.end());

    if (!candidates.empty() && topic_set.count(candidates.front()) > 0) {
        return {candidates.front()};
    }

    std::vector<std::string> fallback;
    for (const auto& topic : candidates) {
        if (topic_set.count(topic) > 0) {
            fallback.push_back(topic);
        }
    }
    return fallback;
}

bool syncVehicleStateNearStamp(RosBagReader& reader,
                               int64_t ref_bag_stamp_ns,
                               int64_t ref_proto_stamp_ns,
                               VehicleStateData& vehicle_state) {
    vehicle_state = VehicleStateData{};
    if (!reader.isOpen()) return false;
    if (ref_proto_stamp_ns <= 0 && ref_bag_stamp_ns <= 0) return false;

    const int64_t bag_anchor =
            ref_bag_stamp_ns > 0 ? ref_bag_stamp_ns : ref_proto_stamp_ns;
    const int64_t car_window_ns = kImageSearchRangeNs / 5;  // 0.2 * 1s
    const bool use_bag_sync =
            ref_bag_stamp_ns > 0 &&
            (ref_proto_stamp_ns <= 0 ||
             timeDiffNs(ref_proto_stamp_ns, ref_bag_stamp_ns) >
                     kBagProtoMismatchNs);
    const int64_t ref_ns = use_bag_sync ? ref_bag_stamp_ns : ref_proto_stamp_ns;

    if (reader.hasTopicTimeIndex()) {
        const auto msg = reader.readMessageNearestTime(
                kCarStateTopic, static_cast<uint64_t>(bag_anchor),
                static_cast<uint64_t>(car_window_ns));
        if (!msg.data.empty()) {
            VehicleStateData state;
            if (ProtoDecoder::decodeVehicleStateFromBag(msg.data, state)) {
                const int64_t msg_ns =
                        use_bag_sync ? static_cast<int64_t>(msg.timestamp_ns)
                                     : state.timestamp_us * 1000;
                if (timeDiffNs(msg_ns, ref_ns) < kImageSyncThresholdNs) {
                    vehicle_state = state;
                    return vehicle_state.has_air_susp_report;
                }
            }
        }
    }

    const uint64_t start_ns = static_cast<uint64_t>(
            std::max<int64_t>(0, bag_anchor - car_window_ns));
    const uint64_t end_ns = static_cast<uint64_t>(bag_anchor + car_window_ns);

    const auto msgs = reader.readAllMessagesParallel({kCarStateTopic}, nullptr,
                                                     start_ns, end_ns);

    int64_t min_delta = INT64_MAX;
    VehicleStateData best;
    for (const auto& msg : msgs) {
        VehicleStateData state;
        if (!ProtoDecoder::decodeVehicleStateFromBag(msg.data, state)) {
            continue;
        }
        const int64_t msg_ns = use_bag_sync
                                       ? static_cast<int64_t>(msg.timestamp_ns)
                                       : state.timestamp_us * 1000;
        const int64_t delta = timeDiffNs(msg_ns, ref_ns);
        if (delta < min_delta) {
            min_delta = delta;
            best = state;
        }
    }

    if (min_delta < kImageSyncThresholdNs) {
        vehicle_state = best;
        return vehicle_state.has_air_susp_report;
    }
    return false;
}

bool decodeCloudMessage(const BagMessage& msg,
                        std::vector<PointXYZIRT>& cloud_raw,
                        int64_t& cloud_stamp_us,
                        std::string& frame_id,
                        const VehicleCalibConfig* calib_config) {
    cloud_raw.clear();
    cloud_stamp_us = 0;
    frame_id.clear();

    ProtoDecoder::PointCloud2Data cloud_data;
    if (!ProtoDecoder::decodePointCloud2FromBag(msg.data, cloud_data)) {
        return false;
    }
    return decombinePointCloud(cloud_data, calib_config, cloud_raw, frame_id,
                               cloud_stamp_us);
}

bool messageNeedsVideoDecode(const BagMessage& msg) {
    if (msg.data.size() < 4) return false;

    uint32_t str_len = 0;
    std::memcpy(&str_len, msg.data.data(), 4);
    if (4 + str_len > msg.data.size()) return false;

    const std::string proto_data =
            ProtoDecoder::stripChurchHeader(msg.data.substr(4, str_len));
    std::string image_buffer;
    std::string format;
    double timestamp_sec = 0;
    if (!ProtoDecoder::decodeCompressedImage(proto_data, image_buffer,
                                             timestamp_sec, &format)) {
        return false;
    }
    if (VideoCodecDecoder::isVideoFormat(format)) return true;
    return format.empty() &&
           VideoCodecDecoder::looksLikeVideoBitstream(
                   reinterpret_cast<const uint8_t*>(image_buffer.data()),
                   image_buffer.size());
}

bool decodeAlignedImageMessages(
        RosBagReader& reader,
        const std::vector<BagMessage>& msgs,
        std::map<std::string, cv::Mat>& images_by_topic,
        std::map<std::string, int64_t>& stamps_ns_by_topic) {
    images_by_topic.clear();
    stamps_ns_by_topic.clear();

    bool need_video = false;
    for (const auto& msg : msgs) {
        if (messageNeedsVideoDecode(msg)) {
            need_video = true;
            break;
        }
    }

    if (!need_video) {
        struct DecodedEntry {
            std::string topic;
            cv::Mat image;
            int64_t stamp_ns = 0;
            bool ok = false;
        };

        std::vector<std::future<DecodedEntry>> futures;
        futures.reserve(msgs.size());
        for (const auto& msg : msgs) {
            if (msg.topic.empty() || msg.data.empty()) continue;
            futures.push_back(
                    std::async(std::launch::async, [msg]() -> DecodedEntry {
                        DecodedEntry entry;
                        entry.topic = msg.topic;
                        double ts_sec = 0;
                        if (!ProtoDecoder::decodeCompressedImageFromBag(
                                    msg.data, entry.image, ts_sec) ||
                            entry.image.empty()) {
                            return entry;
                        }
                        entry.stamp_ns = imageSyncStampNs(msg);
                        entry.ok = true;
                        return entry;
                    }));
        }

        for (auto& fut : futures) {
            DecodedEntry entry = fut.get();
            if (!entry.ok) return false;
            images_by_topic[entry.topic] = std::move(entry.image);
            stamps_ns_by_topic[entry.topic] = entry.stamp_ns;
        }
        return !images_by_topic.empty();
    }

    VideoDecodeCache& cache = reader.videoDecodeCache();
    // Decode topics in parallel — each topic has its own TopicState mutex
    // in VideoDecodeCache, so concurrent decodes for different topics are
    // safe and dramatically reduce per-frame latency for multi-camera BEV.
    struct DecodedVideoEntry {
        std::string topic;
        cv::Mat image;
        int64_t stamp_ns = 0;
        bool ok = false;
    };

    std::vector<std::future<DecodedVideoEntry>> futures;
    futures.reserve(msgs.size());
    for (const auto& msg : msgs) {
        if (msg.topic.empty() || msg.data.empty()) return false;
        futures.push_back(std::async(
                std::launch::async,
                [&reader, &cache, msg]() -> DecodedVideoEntry {
                    DecodedVideoEntry entry;
                    entry.topic = msg.topic;
                    double ts_sec = 0;
                    if (!ProtoDecoder::decodeCompressedImageFromBag(
                                reader, msg, entry.image, ts_sec, cache) ||
                        entry.image.empty()) {
                        return entry;
                    }
                    entry.stamp_ns = imageSyncStampNs(msg);
                    entry.ok = true;
                    return entry;
                }));
    }

    for (auto& fut : futures) {
        DecodedVideoEntry entry = fut.get();
        if (!entry.ok) return false;
        images_by_topic[entry.topic] = std::move(entry.image);
        stamps_ns_by_topic[entry.topic] = entry.stamp_ns;
    }
    return !images_by_topic.empty();
}

bool tryGetAlignedImagesIndexed(
        RosBagReader& reader,
        const std::vector<std::string>& image_topics,
        double percent,
        std::map<std::string, cv::Mat>& images_by_topic,
        std::map<std::string, int64_t>& stamps_ns_by_topic,
        VehicleStateData* vehicle_state_out) {
    if (!reader.hasTopicTimeIndex() || image_topics.empty()) return false;

    // Fast path: try the exact requested percent. Each topic's nearest
    // indexed message is read in parallel and decoded only if all topics
    // sync within 25ms (proto-timestamp based).
    const auto msgs =
            reader.readMessagesAtPercentParallel(image_topics, percent);
    if (msgs.size() == image_topics.size()) {
        std::map<std::string, int64_t> topic_stamps;
        bool all_have_data = true;
        for (const auto& msg : msgs) {
            if (msg.topic.empty() || msg.data.empty()) {
                all_have_data = false;
                break;
            }
            topic_stamps[msg.topic] = imageSyncStampNs(msg);
        }

        if (all_have_data && !topic_stamps.empty()) {
            const int64_t ref_stamp = topic_stamps.begin()->second;
            bool synced = true;
            for (const auto& [topic, stamp] : topic_stamps) {
                (void)topic;
                if (timeDiffNs(stamp, ref_stamp) >= kImageSyncThresholdNs) {
                    synced = false;
                    break;
                }
            }
            if (synced &&
                decodeAlignedImageMessages(reader, msgs, images_by_topic,
                                           stamps_ns_by_topic)) {
                if (vehicle_state_out) {
                    const int64_t ref_bag_stamp =
                            static_cast<int64_t>(msgs.front().timestamp_ns);
                    syncVehicleStateNearStamp(reader, ref_bag_stamp, ref_stamp,
                                              *vehicle_state_out);
                }
                CVLog::Print(
                        "[BagAlignment] getAlignedImages indexed @%.1f%%: "
                        "topics=%zu",
                        percent * 100.0, image_topics.size());
                return true;
            }
        }
    }

    // The exact percent didn't yield a 25ms sync. For HEVC / H.264 bags this
    // would otherwise fall through to tryGetAlignedImagesProtoTimeline, which
    // scans the entire bag (very expensive for multi-bag video sessions).
    //
    // Instead, walk the reference topic's timeline around the requested
    // percent and for each candidate try to assemble a synced group using
    // per-topic time index lookups. Sync is verified on PROTO timestamps
    // (imageSyncStampNs) because bag-record timestamps can drift across
    // cameras/chunks even when the underlying proto ts is identical. If the
    // nearest indexed entry for a topic doesn't proto-sync, we walk forward
    // through that topic's timeline to find one that does — bounded so we
    // don't read the whole bag.
    if (image_topics.size() < 1) return false;
    const std::string ref_topic = pickReferenceImageTopic(image_topics);
    const auto ref_timeline = reader.getMessageTimeline(ref_topic);
    if (ref_timeline.size() < 1) return false;

    // Cache each topic's timeline once — avoids re-querying per attempt.
    std::map<std::string, std::vector<MessageTimeIndexEntry>> topic_timelines;
    for (const auto& topic : image_topics) {
        if (topic == ref_topic) continue;
        topic_timelines[topic] = reader.getMessageTimeline(topic);
        if (topic_timelines[topic].empty()) return false;
    }

    const uint64_t target_ns = reader.percentToTimestampPublic(percent);
    auto ref_cmp = [](const MessageTimeIndexEntry& e, uint64_t t) {
        return e.timestamp_ns < t;
    };
    auto ref_pos = std::lower_bound(ref_timeline.begin(), ref_timeline.end(),
                                    target_ns, ref_cmp);
    size_t center_idx =
            static_cast<size_t>(std::distance(ref_timeline.begin(), ref_pos));
    if (center_idx >= ref_timeline.size()) center_idx = ref_timeline.size() - 1;

    const size_t max_offset =
            std::min<size_t>(40, std::max<size_t>(12, ref_timeline.size() / 4));
    const size_t max_tries =
            std::min<size_t>(2 * max_offset + 1, ref_timeline.size());

    // For non-ref topics, how many neighbours to try when the nearest entry
    // doesn't proto-sync. 3 means: nearest, nearest±1, nearest±2.
    constexpr size_t kNeighbourHops = 3;

    // Phase 1: fast pre-filter — for each candidate idx, use BAG ts (which
    // is what the index stores) to check whether ALL topics have a bag-ts
    // entry within kImageSyncThresholdNs of the ref's bag ts. This avoids
    // the expensive IO+decode for idxs that can't possibly sync.
    // We collect up to kMaxDecodeCandidates promising idxs, then in Phase 2
    // do the full read+decode verification.
    constexpr size_t kMaxDecodeCandidates = 8;
    std::vector<size_t> decode_candidates;
    decode_candidates.reserve(kMaxDecodeCandidates);

    for (size_t attempt = 0; attempt < max_tries; ++attempt) {
        size_t idx = center_idx;
        if (attempt > 0) {
            const size_t offset = (attempt + 1) / 2;
            if (offset > max_offset) break;
            if (attempt % 2 == 1) {
                if (center_idx >= offset) {
                    idx = center_idx - offset;
                } else {
                    continue;
                }
            } else if (center_idx + offset < ref_timeline.size()) {
                idx = center_idx + offset;
            } else {
                continue;
            }
        }

        const uint64_t ref_bag_ts = ref_timeline[idx].timestamp_ns;
        bool bag_synced = true;
        for (const auto& topic : image_topics) {
            if (topic == ref_topic) continue;
            const auto& tl = topic_timelines[topic];
            auto pos =
                    std::lower_bound(tl.begin(), tl.end(), ref_bag_ts, ref_cmp);
            uint64_t best_delta = UINT64_MAX;
            if (pos != tl.end()) {
                best_delta = std::min(best_delta,
                                      pos->timestamp_ns > ref_bag_ts
                                              ? pos->timestamp_ns - ref_bag_ts
                                              : ref_bag_ts - pos->timestamp_ns);
            }
            if (pos != tl.begin()) {
                auto prev = pos - 1;
                best_delta = std::min(
                        best_delta, ref_bag_ts > prev->timestamp_ns
                                            ? ref_bag_ts - prev->timestamp_ns
                                            : prev->timestamp_ns - ref_bag_ts);
            }
            if (best_delta > kImageSyncThresholdNs) {
                bag_synced = false;
                break;
            }
        }
        if (bag_synced) {
            decode_candidates.push_back(idx);
            if (decode_candidates.size() >= kMaxDecodeCandidates) break;
        }
    }

    // Phase 2: full read+decode verification only for promising candidates.
    // This avoids the O(N_attempts * N_topics) IO+decode that dominated
    // latency when many idxs failed HEVC decode (e.g. bag starts mid-GOP
    // and the first sync frame is dozens of frames in).
    for (size_t idx : decode_candidates) {
        BagMessage ref_msg =
                reader.readMessageAtIndexEntry(ref_topic, ref_timeline[idx]);
        if (ref_msg.data.empty()) continue;
        if (ref_msg.topic.empty()) ref_msg.topic = ref_topic;
        const int64_t ref_proto_stamp = imageSyncStampNs(ref_msg);
        if (ref_proto_stamp <= 0) continue;
        const int64_t ref_bag_stamp =
                static_cast<int64_t>(ref_msg.timestamp_ns);

        std::vector<BagMessage> synced_msgs;
        synced_msgs.reserve(image_topics.size());
        synced_msgs.push_back(std::move(ref_msg));
        bool all_synced = true;

        for (const auto& topic : image_topics) {
            if (topic == ref_topic) continue;
            const auto& tl = topic_timelines[topic];
            if (tl.empty()) {
                all_synced = false;
                break;
            }

            auto pos = std::lower_bound(tl.begin(), tl.end(), ref_bag_stamp,
                                        ref_cmp);
            const size_t pos_idx =
                    static_cast<size_t>(std::distance(tl.begin(), pos));

            bool topic_synced = false;
            for (size_t hop = 0; hop <= 2 * kNeighbourHops; ++hop) {
                int64_t cand_idx_signed;
                if (hop == 0) {
                    cand_idx_signed = static_cast<int64_t>(pos_idx);
                } else if (hop % 2 == 1) {
                    cand_idx_signed = static_cast<int64_t>(pos_idx) -
                                      static_cast<int64_t>((hop + 1) / 2);
                } else {
                    cand_idx_signed = static_cast<int64_t>(pos_idx) +
                                      static_cast<int64_t>(hop / 2);
                }
                if (cand_idx_signed < 0 ||
                    cand_idx_signed >= static_cast<int64_t>(tl.size())) {
                    continue;
                }
                const size_t cand_idx = static_cast<size_t>(cand_idx_signed);

                BagMessage msg =
                        reader.readMessageAtIndexEntry(topic, tl[cand_idx]);
                if (msg.data.empty()) continue;
                if (msg.topic.empty()) msg.topic = topic;

                const int64_t stamp = imageSyncStampNs(msg);
                if (stamp <= 0) continue;
                if (timeDiffNs(stamp, ref_proto_stamp) >=
                    kImageSyncThresholdNs) {
                    continue;
                }
                synced_msgs.push_back(std::move(msg));
                topic_synced = true;
                break;
            }

            if (!topic_synced) {
                all_synced = false;
                break;
            }
        }

        if (!all_synced || synced_msgs.size() != image_topics.size()) continue;

        if (!decodeAlignedImageMessages(reader, synced_msgs, images_by_topic,
                                        stamps_ns_by_topic)) {
            continue;
        }

        if (vehicle_state_out) {
            syncVehicleStateNearStamp(reader, ref_bag_stamp, ref_proto_stamp,
                                      *vehicle_state_out);
        }
        CVLog::Print(
                "[BagAlignment] getAlignedImages indexed scan @%.1f%%: "
                "topics=%zu ref=%s idx=%zu/%zu proto=%.3f",
                percent * 100.0, image_topics.size(), ref_topic.c_str(), idx,
                ref_timeline.size(),
                static_cast<double>(ref_proto_stamp) / 1e9);
        return true;
    }

    return false;
}

std::string pickReferenceImageTopic(
        const std::vector<std::string>& image_topics) {
    for (const auto& topic : image_topics) {
        if (topic.find("camera_1") != std::string::npos ||
            topic.find("panoramic_1") != std::string::npos) {
            return topic;
        }
    }
    std::vector<std::string> sorted = image_topics;
    std::sort(sorted.begin(), sorted.end());
    return sorted.front();
}

bool decodeSyncedImagePointers(
        RosBagReader& reader,
        const std::map<std::string, const BagMessage*>& corr_msgs,
        std::map<std::string, cv::Mat>& images_by_topic,
        std::map<std::string, int64_t>& stamps_ns_by_topic) {
    images_by_topic.clear();
    stamps_ns_by_topic.clear();
    if (corr_msgs.empty()) return false;

    bool need_video = false;
    for (const auto& [topic, msg_ptr] : corr_msgs) {
        (void)topic;
        if (msg_ptr && messageNeedsVideoDecode(*msg_ptr)) {
            need_video = true;
            break;
        }
    }

    if (!need_video) {
        struct DecodedEntry {
            std::string topic;
            cv::Mat image;
            int64_t stamp_ns = 0;
            bool ok = false;
        };

        std::vector<std::future<DecodedEntry>> futures;
        futures.reserve(corr_msgs.size());
        for (const auto& [topic, msg_ptr] : corr_msgs) {
            if (!msg_ptr) return false;
            const BagMessage msg = *msg_ptr;
            futures.push_back(std::async(
                    std::launch::async, [topic, msg]() -> DecodedEntry {
                        DecodedEntry entry;
                        entry.topic = topic;
                        double ts_sec = 0;
                        if (!ProtoDecoder::decodeCompressedImageFromBag(
                                    msg.data, entry.image, ts_sec) ||
                            entry.image.empty()) {
                            return entry;
                        }
                        entry.stamp_ns = imageSyncStampNs(msg);
                        entry.ok = true;
                        return entry;
                    }));
        }

        for (auto& fut : futures) {
            DecodedEntry entry = fut.get();
            if (!entry.ok) return false;
            images_by_topic[entry.topic] = std::move(entry.image);
            stamps_ns_by_topic[entry.topic] = entry.stamp_ns;
        }
        return images_by_topic.size() == corr_msgs.size();
    }

    std::vector<BagMessage> msgs;
    msgs.reserve(corr_msgs.size());
    for (const auto& [topic, msg_ptr] : corr_msgs) {
        if (!msg_ptr) return false;
        msgs.push_back(*msg_ptr);
        msgs.back().topic = topic;
    }
    return decodeAlignedImageMessages(reader, msgs, images_by_topic,
                                      stamps_ns_by_topic);
}

bool findSyncedImagePointersNearProto(
        const std::map<std::string,
                       std::vector<std::pair<int64_t, BagMessage>>>&
                topic_frames,
        const std::vector<std::string>& image_topics,
        int64_t ref_proto_ns,
        std::map<std::string, const BagMessage*>& corr_msgs) {
    corr_msgs.clear();
    if (ref_proto_ns <= 0) return false;

    for (const auto& topic : image_topics) {
        auto it = topic_frames.find(topic);
        if (it == topic_frames.end() || it->second.empty()) {
            return false;
        }

        int64_t min_delta = INT64_MAX;
        const BagMessage* best = nullptr;
        for (const auto& [stamp_ns, bag_msg] : it->second) {
            const int64_t delta = timeDiffNs(stamp_ns, ref_proto_ns);
            if (delta < min_delta) {
                min_delta = delta;
                best = &bag_msg;
            }
        }
        if (min_delta >= kImageSyncThresholdNs || best == nullptr) {
            return false;
        }
        corr_msgs[topic] = best;
    }
    return corr_msgs.size() == image_topics.size();
}

bool tryGetAlignedImagesProtoTimeline(
        RosBagReader& reader,
        const std::vector<std::string>& image_topics,
        double percent,
        std::map<std::string, cv::Mat>& images_by_topic,
        std::map<std::string, int64_t>& stamps_ns_by_topic,
        VehicleStateData* vehicle_state_out) {
    if (!reader.isOpen() || image_topics.size() < 2) return false;

    const std::string ref_topic = pickReferenceImageTopic(image_topics);
    std::vector<int64_t> ref_timeline;
    ref_timeline.reserve(512);
    std::set<std::string> ref_only{ref_topic};
    reader.readMessages(
            [&](const BagMessage& msg) {
                ref_timeline.push_back(imageSyncStampNs(msg));
                return true;
            },
            ref_only, reader.getBeginTime(), reader.getEndTime());

    if (ref_timeline.empty()) return false;
    std::sort(ref_timeline.begin(), ref_timeline.end());
    ref_timeline.erase(std::unique(ref_timeline.begin(), ref_timeline.end()),
                       ref_timeline.end());

    const size_t center_idx = std::min(
            ref_timeline.size() - 1,
            static_cast<size_t>(
                    percent * static_cast<double>(ref_timeline.size() - 1) +
                    0.5));

    std::set<std::string> topic_set(image_topics.begin(), image_topics.end());
    std::map<std::string, std::vector<std::pair<int64_t, BagMessage>>>
            topic_frames;
    reader.readMessages(
            [&](const BagMessage& msg) {
                topic_frames[msg.topic].emplace_back(imageSyncStampNs(msg),
                                                     msg);
                return true;
            },
            topic_set, reader.getBeginTime(), reader.getEndTime());

    if (topic_frames.size() != image_topics.size()) {
        CVLog::Warning("[BagAlignment] proto timeline: missing topics %zu/%zu",
                       topic_frames.size(), image_topics.size());
        return false;
    }

    const size_t max_offset =
            std::min<size_t>(40, std::max<size_t>(8, ref_timeline.size() / 20));
    const size_t max_tries =
            std::min<size_t>(2 * max_offset + 1, ref_timeline.size());
    for (size_t attempt = 0; attempt < max_tries; ++attempt) {
        size_t idx = center_idx;
        if (attempt > 0) {
            const size_t offset = (attempt + 1) / 2;
            if (offset > max_offset) {
                break;
            }
            if (attempt % 2 == 1) {
                if (center_idx >= offset) {
                    idx = center_idx - offset;
                } else {
                    continue;
                }
            } else if (center_idx + offset < ref_timeline.size()) {
                idx = center_idx + offset;
            } else {
                continue;
            }
        }

        const int64_t ref_proto = ref_timeline[idx];
        std::map<std::string, const BagMessage*> corr_msgs;
        if (!findSyncedImagePointersNearProto(topic_frames, image_topics,
                                              ref_proto, corr_msgs)) {
            continue;
        }
        if (!decodeSyncedImagePointers(reader, corr_msgs, images_by_topic,
                                       stamps_ns_by_topic)) {
            continue;
        }

        if (vehicle_state_out) {
            int64_t ref_bag_stamp_ns = 0;
            for (const auto& [topic, msg_ptr] : corr_msgs) {
                (void)topic;
                ref_bag_stamp_ns = static_cast<int64_t>(msg_ptr->timestamp_ns);
                break;
            }
            syncVehicleStateNearStamp(reader, ref_bag_stamp_ns, ref_proto,
                                      *vehicle_state_out);
        }
        CVLog::Print(
                "[BagAlignment] getAlignedImages proto timeline: topics=%zu "
                "ref=%s idx=%zu/%zu proto=%.3f delta_ms=%.1f",
                image_topics.size(), ref_topic.c_str(), idx,
                ref_timeline.size(), static_cast<double>(ref_proto) / 1e9,
                timeDiffNs(stamps_ns_by_topic.begin()->second, ref_proto) /
                        1e6);
        return true;
    }

    CVLog::Warning(
            "[BagAlignment] proto timeline sync failed: topics=%zu ref=%s "
            "timeline=%zu center=%.1f%%",
            image_topics.size(), ref_topic.c_str(), ref_timeline.size(),
            percent * 100.0);
    return false;
}

bool findCloudNearRefInFullBag(RosBagReader& reader,
                               const std::vector<std::string>& cloud_topics,
                               int64_t ref_stamp_ns,
                               bool allow_cloud_as_ref,
                               std::vector<PointXYZIRT>& cloud_raw,
                               int64_t& cloud_stamp_us,
                               std::string* frame_id_out,
                               int64_t ref_bag_stamp_ns,
                               const VehicleCalibConfig* calib_config) {
    cloud_raw.clear();
    cloud_stamp_us = 0;
    if (frame_id_out) frame_id_out->clear();
    if (!reader.isOpen()) return false;

    const auto active_topics = selectCloudTopics(reader, cloud_topics);
    if (active_topics.empty()) return false;

    // codetree get_aligned_images_cloud scans the full bag time span.
    std::set<std::string> topic_set(active_topics.begin(), active_topics.end());
    const auto msgs = reader.readAllMessagesParallel(
            topic_set, nullptr, reader.getBeginTime(), reader.getEndTime());
    if (msgs.empty() && !allow_cloud_as_ref) {
        return false;
    }

    int64_t ref_stamp = ref_stamp_ns;
    int64_t min_delta = INT64_MAX;
    const BagMessage* best_msg = nullptr;

    for (const auto& msg : msgs) {
        int64_t delta = INT64_MAX;
        if (ref_bag_stamp_ns > 0) {
            delta = timeDiffNs(static_cast<int64_t>(msg.timestamp_ns),
                               ref_bag_stamp_ns);
        } else {
            double ts_sec = 0;
            if (!ProtoDecoder::extractPointCloudTimestampFromBag(msg.data,
                                                                 ts_sec)) {
                continue;
            }
            const int64_t stamp_ns = static_cast<int64_t>(ts_sec * 1e9);
            if (ref_stamp <= 0 && allow_cloud_as_ref) {
                ref_stamp = stamp_ns;
            }
            delta = timeDiffNs(stamp_ns, ref_stamp);
        }
        if (delta < min_delta) {
            min_delta = delta;
            best_msg = &msg;
        }
    }

    if (best_msg == nullptr || min_delta >= kCloudSyncThresholdNs) {
        CVLog::Warning(
                "[BagAlignment] cloud full-bag sync failed: min_delta=%.3f ms",
                min_delta / 1e6);
        return false;
    }

    std::string frame_id;
    if (!decodeCloudMessage(*best_msg, cloud_raw, cloud_stamp_us, frame_id,
                            calib_config)) {
        return false;
    }
    if (frame_id_out) {
        *frame_id_out = frame_id;
    }
    CVLog::Print(
            "[BagAlignment] cloud full-bag aligned: delta=%.3f ms, points=%zu",
            min_delta / 1e6, cloud_raw.size());
    return true;
}

}  // namespace

bool bagUsesVideoCodec(BagImageEncoding encoding) {
    return encoding == BagImageEncoding::Hevc ||
           encoding == BagImageEncoding::H264 ||
           encoding == BagImageEncoding::Mixed;
}

BagImageEncoding probeBagImageEncoding(
        RosBagReader& reader, const std::vector<std::string>& camera_topics) {
    if (!reader.isOpen() || camera_topics.empty()) {
        return BagImageEncoding::Unknown;
    }

    bool has_jpeg = false;
    bool has_h264 = false;
    bool has_hevc = false;

    for (const auto& topic : camera_topics) {
        const auto msg = reader.readFirstMessage(topic);
        if (msg.data.empty()) continue;
        if (!messageNeedsVideoDecode(msg)) {
            has_jpeg = true;
            continue;
        }

        uint32_t str_len = 0;
        std::memcpy(&str_len, msg.data.data(), 4);
        if (4 + str_len > msg.data.size()) continue;
        const std::string proto_data =
                ProtoDecoder::stripChurchHeader(msg.data.substr(4, str_len));
        std::string format;
        std::string image_buffer;
        double ts_sec = 0;
        if (!ProtoDecoder::decodeCompressedImage(proto_data, image_buffer,
                                                 ts_sec, &format)) {
            continue;
        }
        if (VideoCodecDecoder::codecFromFormat(format) ==
            VideoCodecDecoder::Codec::HEVC) {
            has_hevc = true;
        } else {
            has_h264 = true;
        }
    }

    if ((has_hevc || has_h264) && has_jpeg) return BagImageEncoding::Mixed;
    if (has_hevc) return BagImageEncoding::Hevc;
    if (has_h264) return BagImageEncoding::H264;
    if (has_jpeg) return BagImageEncoding::Jpeg;
    return BagImageEncoding::Unknown;
}

bool getAlignedImages(RosBagReader& reader,
                      const std::vector<std::string>& image_topics,
                      double percent,
                      std::map<std::string, cv::Mat>& images_by_topic,
                      std::map<std::string, int64_t>& stamps_ns_by_topic,
                      VehicleStateData* vehicle_state_out) {
    images_by_topic.clear();
    stamps_ns_by_topic.clear();
    if (vehicle_state_out) {
        *vehicle_state_out = VehicleStateData{};
    }
    if (!reader.isOpen() || image_topics.empty()) return false;

    if (tryGetAlignedImagesIndexed(reader, image_topics, percent,
                                   images_by_topic, stamps_ns_by_topic,
                                   vehicle_state_out)) {
        return true;
    }

    uint64_t start_ns = 0;
    uint64_t end_ns = UINT64_MAX;
    computeImageSearchWindow(reader, percent, start_ns, end_ns);

    std::set<std::string> topic_set(image_topics.begin(), image_topics.end());
    const auto msgs = reader.readAllMessagesParallel(topic_set, nullptr,
                                                     start_ns, end_ns);

    const std::string ref_topic = pickReferenceImageTopic(image_topics);

    CVLog::Print(
            "[BagAlignment] getAlignedImages: window=[%llu,%llu] ns, "
            "topics=%zu, raw_msgs=%zu",
            static_cast<unsigned long long>(start_ns),
            static_cast<unsigned long long>(end_ns), image_topics.size(),
            msgs.size());

    std::map<std::string, std::map<int64_t, BagMessage>> image_msgs;
    for (const auto& msg : msgs) {
        const int64_t stamp_ns = imageSyncStampNs(msg);
        if (stamp_ns <= 0) continue;
        image_msgs[msg.topic][stamp_ns] = msg;
    }

    if (image_msgs.empty()) {
        CVLog::Warning("[BagAlignment] no image messages in search window");
        return false;
    }

    auto ref_it = image_msgs.find(ref_topic);
    if (ref_it == image_msgs.end()) {
        ref_it = image_msgs.begin();
    }
    const auto& ref_msgs = ref_it->second;
    size_t ref_tried = 0;
    for (const auto& [ref_stamp, ref_msg] : ref_msgs) {
        (void)ref_msg;
        ++ref_tried;
        std::map<std::string, const BagMessage*> corr_msgs;
        bool all_synced = true;
        for (const auto& [topic, topic_msgs] : image_msgs) {
            int64_t min_delta = INT64_MAX;
            const BagMessage* best = nullptr;
            for (const auto& [stamp, bag_msg] : topic_msgs) {
                const int64_t delta = timeDiffNs(stamp, ref_stamp);
                if (delta < min_delta) {
                    min_delta = delta;
                    best = &bag_msg;
                }
            }
            if (min_delta < kImageSyncThresholdNs && best != nullptr) {
                corr_msgs[topic] = best;
            } else {
                all_synced = false;
                break;
            }
        }

        if (!all_synced || corr_msgs.size() != image_msgs.size()) {
            continue;
        }

        if (vehicle_state_out) {
            int64_t ref_bag_stamp_ns = 0;
            for (const auto& [topic, msg_ptr] : corr_msgs) {
                (void)topic;
                ref_bag_stamp_ns = static_cast<int64_t>(msg_ptr->timestamp_ns);
                break;
            }
            syncVehicleStateNearStamp(reader, ref_bag_stamp_ns, ref_stamp,
                                      *vehicle_state_out);
        }

        struct DecodedEntry {
            std::string topic;
            cv::Mat image;
            int64_t stamp_ns = 0;
            bool ok = false;
        };

        bool need_video = false;
        for (const auto& [topic, msg_ptr] : corr_msgs) {
            (void)topic;
            if (msg_ptr && messageNeedsVideoDecode(*msg_ptr)) {
                need_video = true;
                break;
            }
        }

        std::map<std::string, cv::Mat> decoded_images;
        std::map<std::string, int64_t> decoded_stamps;
        if (!need_video) {
            std::vector<std::pair<std::string, const BagMessage*>> msg_list;
            msg_list.reserve(corr_msgs.size());
            for (const auto& [topic, msg_ptr] : corr_msgs) {
                msg_list.emplace_back(topic, msg_ptr);
            }

            std::vector<std::future<DecodedEntry>> futures;
            futures.reserve(msg_list.size());
            for (const auto& [topic, msg_ptr] : msg_list) {
                futures.push_back(std::async(
                        std::launch::async, [topic, msg_ptr]() -> DecodedEntry {
                            DecodedEntry entry;
                            entry.topic = topic;
                            double ts_sec = 0;
                            if (!ProtoDecoder::decodeCompressedImageFromBag(
                                        msg_ptr->data, entry.image, ts_sec) ||
                                entry.image.empty()) {
                                return entry;
                            }
                            entry.stamp_ns = imageSyncStampNs(*msg_ptr);
                            entry.ok = true;
                            return entry;
                        }));
            }

            for (auto& fut : futures) {
                DecodedEntry entry = fut.get();
                if (!entry.ok) {
                    all_synced = false;
                    break;
                }
                decoded_images[entry.topic] = std::move(entry.image);
                decoded_stamps[entry.topic] = entry.stamp_ns;
            }
        } else {
            std::vector<BagMessage> sync_msgs;
            sync_msgs.reserve(corr_msgs.size());
            for (const auto& [topic, msg_ptr] : corr_msgs) {
                if (!msg_ptr) {
                    all_synced = false;
                    break;
                }
                sync_msgs.push_back(*msg_ptr);
                sync_msgs.back().topic = topic;
            }
            if (!all_synced ||
                !decodeAlignedImageMessages(reader, sync_msgs, decoded_images,
                                            decoded_stamps)) {
                all_synced = false;
            }
        }

        if (!all_synced || decoded_images.empty()) {
            continue;
        }

        images_by_topic = std::move(decoded_images);
        stamps_ns_by_topic = std::move(decoded_stamps);
        CVLog::Print(
                "[BagAlignment] getAlignedImages synced: ref_topic=%s "
                "topics=%zu "
                "window=[%llu,%llu]",
                ref_topic.c_str(), image_topics.size(),
                static_cast<unsigned long long>(start_ns),
                static_cast<unsigned long long>(end_ns));
        return true;
    }

    if (image_topics.size() > 1 &&
        tryGetAlignedImagesProtoTimeline(reader, image_topics, percent,
                                         images_by_topic, stamps_ns_by_topic,
                                         vehicle_state_out)) {
        CVLog::Warning(
                "[BagAlignment] getAlignedImages: bag-window sync failed, "
                "used proto-timeline fallback");
        return true;
    }

    CVLog::Warning(
            "[BagAlignment] image sync failed: ref_topic=%s, topics=%zu, "
            "ref_candidates=%zu (25ms threshold)",
            ref_topic.c_str(), image_msgs.size(), ref_tried);
    return false;
}

bool getAlignedCloud(RosBagReader& reader,
                     const std::vector<std::string>& cloud_topics,
                     int64_t ref_stamp_ns,
                     bool allow_cloud_as_ref,
                     std::vector<PointXYZIRT>& cloud_raw,
                     int64_t& cloud_stamp_us,
                     std::string* frame_id_out,
                     int64_t ref_bag_stamp_ns,
                     const VehicleCalibConfig* calib_config) {
    cloud_raw.clear();
    cloud_stamp_us = 0;
    if (frame_id_out) frame_id_out->clear();
    if (!reader.isOpen()) return false;
    if (ref_stamp_ns <= 0 && ref_bag_stamp_ns <= 0 && !allow_cloud_as_ref) {
        return false;
    }

    const auto active_topics = selectCloudTopics(reader, cloud_topics);
    if (active_topics.empty()) return false;

    if (reader.hasTopicTimeIndex()) {
        int64_t ref_stamp = ref_stamp_ns;
        if (ref_stamp == 0 && allow_cloud_as_ref) {
            ref_stamp = static_cast<int64_t>(reader.getBeginTime());
        }
        const uint64_t lookup_bag_ns =
                (ref_bag_stamp_ns > 0)
                        ? static_cast<uint64_t>(ref_bag_stamp_ns)
                        : static_cast<uint64_t>(
                                  std::max<int64_t>(0, ref_stamp));
        for (const auto& topic : active_topics) {
            const auto msg = reader.readMessageNearestTime(
                    topic, lookup_bag_ns,
                    static_cast<uint64_t>(kAlignSearchRangeNs));
            if (msg.data.empty()) continue;

            if (ref_bag_stamp_ns > 0) {
                if (timeDiffNs(static_cast<int64_t>(msg.timestamp_ns),
                               ref_bag_stamp_ns) >= kCloudSyncThresholdNs) {
                    continue;
                }
            } else if (ref_stamp > 0 &&
                       timeDiffNs(cloudSyncStampNs(msg), ref_stamp) >=
                               kCloudSyncThresholdNs) {
                continue;
            }
            std::string frame_id;
            if (!decodeCloudMessage(msg, cloud_raw, cloud_stamp_us, frame_id,
                                    calib_config)) {
                continue;
            }
            if (frame_id_out) {
                *frame_id_out = frame_id;
            }
            CVLog::Print(
                    "[BagAlignment] cloud aligned (indexed): delta=%.3f ms, "
                    "points=%zu",
                    timeDiffNs(static_cast<int64_t>(msg.timestamp_ns),
                               ref_bag_stamp_ns > 0 ? ref_bag_stamp_ns
                                                    : ref_stamp) /
                            1e6,
                    cloud_raw.size());
            return true;
        }
    }

    std::set<std::string> topic_set(active_topics.begin(), active_topics.end());
    const auto msgs = reader.readAllMessagesParallel(
            topic_set, nullptr, reader.getBeginTime(), reader.getEndTime());

    int64_t min_delta = INT64_MAX;
    const BagMessage* best_msg = nullptr;
    int64_t best_stamp_ns = 0;

    for (const auto& msg : msgs) {
        int64_t ref_stamp = ref_stamp_ns;
        int64_t delta = INT64_MAX;
        if (ref_bag_stamp_ns > 0) {
            delta = timeDiffNs(static_cast<int64_t>(msg.timestamp_ns),
                               ref_bag_stamp_ns);
        } else {
            const int64_t stamp_ns = cloudSyncStampNs(msg);
            if (ref_stamp <= 0 && allow_cloud_as_ref) {
                ref_stamp = stamp_ns;
            }
            delta = timeDiffNs(stamp_ns, ref_stamp);
        }
        if (delta < min_delta) {
            min_delta = delta;
            best_msg = &msg;
            best_stamp_ns = cloudSyncStampNs(msg);
        }
    }

    if (best_msg == nullptr || min_delta >= kCloudSyncThresholdNs) {
        CVLog::Warning("[BagAlignment] cloud sync failed: min_delta=%.3f ms",
                       min_delta / 1e6);
        return false;
    }

    std::string frame_id;
    if (!decodeCloudMessage(*best_msg, cloud_raw, cloud_stamp_us, frame_id,
                            calib_config)) {
        return false;
    }
    if (frame_id_out) {
        *frame_id_out = frame_id;
    }
    CVLog::Print(
            "[BagAlignment] cloud aligned: delta=%.3f ms, points=%zu, frame=%s",
            timeDiffNs(best_stamp_ns, ref_stamp_ns) / 1e6, cloud_raw.size(),
            frame_id.c_str());
    return true;
}

bool loadCloudNearImageStamp(RosBagReader& reader,
                             const std::vector<std::string>& cloud_topics,
                             int64_t ref_stamp_ns,
                             std::vector<PointXYZIRT>& cloud_raw,
                             int64_t& cloud_stamp_us,
                             std::string* frame_id_out,
                             int64_t ref_bag_stamp_ns,
                             const VehicleCalibConfig* calib_config) {
    cloud_raw.clear();
    cloud_stamp_us = 0;
    if (frame_id_out) frame_id_out->clear();
    if (!reader.isOpen() || ref_stamp_ns <= 0) return false;

    const auto active_topics = selectCloudTopics(reader, cloud_topics);
    if (active_topics.empty()) return false;

    if (reader.hasTopicTimeIndex()) {
        const uint64_t lookup_bag_ns =
                (ref_bag_stamp_ns > 0)
                        ? static_cast<uint64_t>(ref_bag_stamp_ns)
                        : static_cast<uint64_t>(
                                  std::max<int64_t>(0, ref_stamp_ns));
        for (const auto& topic : active_topics) {
            const auto msg = reader.readMessageNearestTime(
                    topic, lookup_bag_ns,
                    static_cast<uint64_t>(kAlignSearchRangeNs));
            if (msg.data.empty()) continue;

            if (ref_bag_stamp_ns > 0) {
                if (timeDiffNs(static_cast<int64_t>(msg.timestamp_ns),
                               ref_bag_stamp_ns) >= kCloudSyncThresholdNs) {
                    continue;
                }
            } else {
                const int64_t stamp_ns = cloudSyncStampNs(msg);
                if (timeDiffNs(stamp_ns, ref_stamp_ns) >=
                    kCloudSyncThresholdNs) {
                    continue;
                }
            }

            std::string frame_id;
            if (!decodeCloudMessage(msg, cloud_raw, cloud_stamp_us, frame_id,
                                    calib_config)) {
                continue;
            }
            if (frame_id_out) {
                *frame_id_out = frame_id;
            }
            CVLog::Print(
                    "[BagAlignment] loadCloudNearImageStamp (indexed): %zu "
                    "points",
                    cloud_raw.size());
            return true;
        }
        CVLog::Warning(
                "[BagAlignment] loadCloudNearImageStamp indexed lookup failed, "
                "trying window scan");
    }

    const uint64_t start_ns =
            (ref_bag_stamp_ns > 0)
                    ? static_cast<uint64_t>(std::max<int64_t>(
                              0, ref_bag_stamp_ns - kAlignSearchRangeNs))
                    : static_cast<uint64_t>(std::max<int64_t>(
                              0, ref_stamp_ns - kAlignSearchRangeNs));
    const uint64_t end_ns =
            (ref_bag_stamp_ns > 0)
                    ? static_cast<uint64_t>(ref_bag_stamp_ns +
                                            kAlignSearchRangeNs)
                    : static_cast<uint64_t>(ref_stamp_ns + kAlignSearchRangeNs);

    std::set<std::string> topic_set(active_topics.begin(), active_topics.end());
    const auto msgs = reader.readAllMessagesParallel(topic_set, nullptr,
                                                     start_ns, end_ns);

    int64_t min_delta = INT64_MAX;
    const BagMessage* best_msg = nullptr;
    for (const auto& msg : msgs) {
        int64_t delta = INT64_MAX;
        if (ref_bag_stamp_ns > 0) {
            delta = timeDiffNs(static_cast<int64_t>(msg.timestamp_ns),
                               ref_bag_stamp_ns);
        } else {
            const int64_t stamp_ns = cloudSyncStampNs(msg);
            delta = timeDiffNs(stamp_ns, ref_stamp_ns);
        }
        if (delta < min_delta) {
            min_delta = delta;
            best_msg = &msg;
        }
    }

    if (best_msg == nullptr || min_delta >= kCloudSyncThresholdNs) {
        CVLog::Warning(
                "[BagAlignment] loadCloudNearImageStamp failed: min_delta=%.3f "
                "ms",
                min_delta / 1e6);
        return false;
    }

    std::string frame_id;
    if (!decodeCloudMessage(*best_msg, cloud_raw, cloud_stamp_us, frame_id,
                            calib_config)) {
        return false;
    }
    if (frame_id_out) {
        *frame_id_out = frame_id;
    }
    CVLog::Print("[BagAlignment] loadCloudNearImageStamp: %zu points, frame=%s",
                 cloud_raw.size(), frame_id.c_str());
    return true;
}

void partitionCameraImageTopics(const std::vector<std::string>& image_topics,
                                std::vector<std::string>& svm_out,
                                std::vector<std::string>& avm_out) {
    svm_out.clear();
    avm_out.clear();
    for (const auto& topic : image_topics) {
        if (topic.find("panoramic_") != std::string::npos) {
            avm_out.push_back(topic);
        } else if (topic.find("/sensors/camera/") != std::string::npos) {
            svm_out.push_back(topic);
        }
    }
    std::sort(svm_out.begin(), svm_out.end());
    std::sort(avm_out.begin(), avm_out.end());
}

namespace {

bool getAlignedImagesWithBevGroupsIfNeeded(
        RosBagReader& reader,
        const std::vector<std::string>& image_topics,
        double percent,
        std::map<std::string, cv::Mat>& images_by_topic,
        std::map<std::string, int64_t>& stamps_ns_by_topic,
        VehicleStateData* vehicle_state_out) {
    std::vector<std::string> svm_topics;
    std::vector<std::string> avm_topics;
    partitionCameraImageTopics(image_topics, svm_topics, avm_topics);
    const bool use_bev_groups =
            !svm_topics.empty() && !avm_topics.empty() &&
            svm_topics.size() + avm_topics.size() == image_topics.size();
    if (use_bev_groups) {
        return getAlignedImagesBevGroups(reader, svm_topics, avm_topics,
                                         percent, images_by_topic,
                                         stamps_ns_by_topic, vehicle_state_out);
    }
    return getAlignedImages(reader, image_topics, percent, images_by_topic,
                            stamps_ns_by_topic, vehicle_state_out);
}

}  // namespace

bool getAlignedImagesCloud(RosBagReader& reader,
                           const std::vector<std::string>& image_topics,
                           const std::vector<std::string>& cloud_topics,
                           double percent,
                           bool allow_cloud_as_ref,
                           std::map<std::string, cv::Mat>& images_by_topic,
                           std::map<std::string, int64_t>& stamps_ns_by_topic,
                           std::vector<PointXYZIRT>& cloud_raw,
                           int64_t& cloud_stamp_us,
                           std::string* frame_id_out,
                           VehicleStateData* vehicle_state_out,
                           const VehicleCalibConfig* calib_config) {
    cloud_raw.clear();
    cloud_stamp_us = 0;

    const bool has_images = getAlignedImagesWithBevGroupsIfNeeded(
            reader, image_topics, percent, images_by_topic, stamps_ns_by_topic,
            vehicle_state_out);

    int64_t ref_stamp_ns = 0;
    int64_t ref_bag_stamp_ns = 0;
    if (has_images && !stamps_ns_by_topic.empty()) {
        const std::string ref_topic = pickReferenceImageTopic(image_topics);
        auto it = stamps_ns_by_topic.find(ref_topic);
        if (it == stamps_ns_by_topic.end()) {
            it = stamps_ns_by_topic.begin();
        }
        if (it != stamps_ns_by_topic.end()) {
            ref_stamp_ns = it->second;
            ref_bag_stamp_ns = 0;
            const auto cam_msg = reader.readMessageNearestTime(
                    ref_topic, static_cast<uint64_t>(ref_stamp_ns),
                    static_cast<uint64_t>(kAlignSearchRangeNs));
            if (!cam_msg.data.empty()) {
                ref_bag_stamp_ns = static_cast<int64_t>(cam_msg.timestamp_ns);
            }
        }
    }

    std::string frame_id;
    bool has_cloud = getAlignedCloud(reader, cloud_topics, ref_stamp_ns,
                                     allow_cloud_as_ref && !has_images,
                                     cloud_raw, cloud_stamp_us, &frame_id,
                                     ref_bag_stamp_ns, calib_config);
    if (!has_cloud) {
        has_cloud = findCloudNearRefInFullBag(
                reader, cloud_topics, ref_stamp_ns,
                allow_cloud_as_ref && !has_images, cloud_raw, cloud_stamp_us,
                &frame_id, ref_bag_stamp_ns, calib_config);
    }

    if (frame_id_out && has_cloud) {
        *frame_id_out = frame_id;
    }

    if (!has_images && !(allow_cloud_as_ref && has_cloud)) {
        return false;
    }
    return has_images || has_cloud;
}

bool getAlignedImagesForBev(RosBagReader& reader,
                            const std::vector<std::string>& image_topics,
                            double percent,
                            std::map<std::string, cv::Mat>& images_by_topic,
                            std::map<std::string, int64_t>& stamps_ns_by_topic,
                            int64_t& cloud_stamp_ns,
                            VehicleStateData* vehicle_state_out) {
    cloud_stamp_ns = 0;
    if (!getAlignedImagesWithBevGroupsIfNeeded(
                reader, image_topics, percent, images_by_topic,
                stamps_ns_by_topic, vehicle_state_out)) {
        return false;
    }
    if (images_by_topic.size() != image_topics.size()) {
        images_by_topic.clear();
        stamps_ns_by_topic.clear();
        return false;
    }
    if (!stamps_ns_by_topic.empty()) {
        cloud_stamp_ns = stamps_ns_by_topic.begin()->second;
    }
    return true;
}

bool findBestAlignedPercent(RosBagReader& reader,
                            const std::vector<std::string>& image_topics,
                            double search_start_percent,
                            double search_end_percent,
                            double& out_percent,
                            int64_t& out_ref_stamp_ns,
                            double search_step) {
    out_percent = 0.0;
    out_ref_stamp_ns = 0;
    if (!reader.isOpen() || image_topics.empty()) return false;

    const double start = std::max(0.0, search_start_percent);
    const double end = std::min(1.0, search_end_percent);
    if (end <= start) return false;
    const double step = std::max(0.001, search_step);

    for (double percent = start; percent <= end + 1e-9; percent += step) {
        std::map<std::string, cv::Mat> images_by_topic;
        std::map<std::string, int64_t> stamps_ns_by_topic;
        if (!getAlignedImages(reader, image_topics, percent, images_by_topic,
                              stamps_ns_by_topic, nullptr)) {
            continue;
        }
        if (images_by_topic.size() != image_topics.size()) continue;

        out_percent = percent;
        out_ref_stamp_ns = stamps_ns_by_topic.begin()->second;
        CVLog::Print("[BagAlignment] findBestAlignedPercent: %.3f (%zu topics)",
                     percent, image_topics.size());
        return true;
    }

    CVLog::Warning(
            "[BagAlignment] findBestAlignedPercent: no full sync in [%.2f, "
            "%.2f]",
            start, end);
    return false;
}

bool findBestAlignedPercentBevGroups(
        RosBagReader& reader,
        const std::vector<std::string>& svm_image_topics,
        const std::vector<std::string>& avm_image_topics,
        double search_start_percent,
        double search_end_percent,
        double& out_percent,
        int64_t& out_ref_stamp_ns,
        double search_step) {
    out_percent = 0.0;
    out_ref_stamp_ns = 0;
    if (!reader.isOpen()) return false;
    if (svm_image_topics.empty() && avm_image_topics.empty()) return false;

    const double start = std::max(0.0, search_start_percent);
    const double end = std::min(1.0, search_end_percent);
    if (end <= start) return false;
    const double step = std::max(0.001, search_step);

    const size_t expected = svm_image_topics.size() + avm_image_topics.size();

    for (double percent = start; percent <= end + 1e-9; percent += step) {
        std::map<std::string, cv::Mat> images_by_topic;
        std::map<std::string, int64_t> stamps_ns_by_topic;
        if (!getAlignedImagesBevGroups(
                    reader, svm_image_topics, avm_image_topics, percent,
                    images_by_topic, stamps_ns_by_topic, nullptr)) {
            continue;
        }
        if (images_by_topic.size() != expected) continue;

        out_percent = percent;
        out_ref_stamp_ns = stamps_ns_by_topic.begin()->second;
        CVLog::Print(
                "[BagAlignment] findBestAlignedPercentBevGroups: %.3f "
                "(svm=%zu avm=%zu)",
                percent, svm_image_topics.size(), avm_image_topics.size());
        return true;
    }

    CVLog::Warning(
            "[BagAlignment] findBestAlignedPercentBevGroups: no group sync in "
            "[%.2f, %.2f]",
            start, end);
    return false;
}

bool computeBevGroupSliceWindowNs(
        RosBagReader& reader,
        const std::vector<std::string>& svm_image_topics,
        const std::vector<std::string>& avm_image_topics,
        double sync_percent,
        uint64_t pad_ns,
        uint64_t& out_start_ns,
        uint64_t& out_end_ns) {
    out_start_ns = 0;
    out_end_ns = 0;
    if (!reader.isOpen()) return false;

    std::map<std::string, cv::Mat> images_by_topic;
    std::map<std::string, int64_t> proto_stamps_ns;
    if (!getAlignedImagesBevGroups(reader, svm_image_topics, avm_image_topics,
                                   sync_percent, images_by_topic,
                                   proto_stamps_ns, nullptr)) {
        return false;
    }

    const size_t expected = svm_image_topics.size() + avm_image_topics.size();
    if (images_by_topic.size() != expected) return false;

    std::vector<std::string> all_topics;
    all_topics.reserve(expected);
    all_topics.insert(all_topics.end(), svm_image_topics.begin(),
                      svm_image_topics.end());
    all_topics.insert(all_topics.end(), avm_image_topics.begin(),
                      avm_image_topics.end());

    std::set<std::string> topic_set(all_topics.begin(), all_topics.end());
    const auto msgs = reader.readAllMessagesParallel(
            topic_set, nullptr, reader.getBeginTime(), reader.getEndTime());

    std::map<std::string, uint64_t> bag_ts_by_topic;
    for (const auto& msg : msgs) {
        auto stamp_it = proto_stamps_ns.find(msg.topic);
        if (stamp_it == proto_stamps_ns.end()) continue;

        double ts_sec = 0;
        if (!ProtoDecoder::extractCompressedImageTimestampFromBag(msg.data,
                                                                  ts_sec)) {
            continue;
        }
        const int64_t proto_ns = static_cast<int64_t>(ts_sec * 1e9);
        if (timeDiffNs(proto_ns, stamp_it->second) >= kImageSyncThresholdNs) {
            continue;
        }
        bag_ts_by_topic[msg.topic] = msg.timestamp_ns;
    }

    if (bag_ts_by_topic.size() != expected) {
        CVLog::Warning(
                "[BagAlignment] slice window: matched %zu/%zu bag records",
                bag_ts_by_topic.size(), expected);
        return false;
    }

    uint64_t min_ts = UINT64_MAX;
    uint64_t max_ts = 0;
    for (const auto& [topic, bag_ts] : bag_ts_by_topic) {
        (void)topic;
        min_ts = std::min(min_ts, bag_ts);
        max_ts = std::max(max_ts, bag_ts);
    }

    const uint64_t bag_begin = reader.getBeginTime();
    const uint64_t bag_end = reader.getEndTime();
    out_start_ns = (min_ts > pad_ns) ? (min_ts - pad_ns) : bag_begin;
    out_start_ns = std::max(out_start_ns, bag_begin);
    out_end_ns = std::min(max_ts + pad_ns, bag_end);
    if (out_end_ns <= out_start_ns) return false;

    CVLog::Print("[BagAlignment] slice window bag_ts=[%llu,%llu] pad=%.3fs",
                 static_cast<unsigned long long>(out_start_ns),
                 static_cast<unsigned long long>(out_end_ns),
                 static_cast<double>(pad_ns) / 1e9);
    return true;
}

namespace {

struct SelectedBagMessage {
    BagMessage msg;
    int64_t proto_ns = 0;
    int64_t group_ref_proto_ns = 0;
    bool has_proto = false;
};

enum class ExportGroupKind { SVM, AVM, Lidar, Ancillary };

struct ExportGroupSync {
    ExportGroupKind kind = ExportGroupKind::Ancillary;
    double source_percent = 0.0;
    int64_t ref_proto_ns = 0;
    std::map<std::string, int64_t> proto_stamps_ns_by_topic;
};

bool protoInWindow(int64_t proto_ns, int64_t proto_lo, int64_t proto_hi) {
    return proto_ns >= proto_lo && proto_ns <= proto_hi;
}

uint64_t remapProtoNearCenter(int64_t proto_ns,
                              int64_t group_ref_proto_ns,
                              uint64_t out_center_ns) {
    const int64_t mapped = static_cast<int64_t>(out_center_ns) +
                           (proto_ns - group_ref_proto_ns);
    return static_cast<uint64_t>(std::max<int64_t>(0, mapped));
}

uint64_t mapBagToOutputTime(uint64_t bag_ts,
                            uint64_t min_bag_ts,
                            uint64_t bag_span,
                            uint64_t out_base,
                            uint64_t out_duration) {
    if (bag_span == 0) return out_base;
    const uint64_t rel = bag_ts - min_bag_ts;
    const double frac =
            std::max(0.0, std::min(1.0, static_cast<double>(rel) / bag_span));
    return out_base + static_cast<uint64_t>(frac * out_duration);
}

bool collectGroupSyncNear(RosBagReader& reader,
                          const std::vector<std::string>& topics,
                          ExportGroupKind kind,
                          double center_percent,
                          double search_half_width,
                          ExportGroupSync& out_sync) {
    out_sync = ExportGroupSync{};
    out_sync.kind = kind;
    if (!reader.isOpen() || topics.empty()) return false;

    const double start = std::max(0.0, center_percent - search_half_width);
    const double end = std::min(1.0, center_percent + search_half_width);
    if (!findBestAlignedPercent(reader, topics, start, end,
                                out_sync.source_percent,
                                out_sync.ref_proto_ns)) {
        return false;
    }

    std::map<std::string, cv::Mat> images_by_topic;
    if (!getAlignedImages(reader, topics, out_sync.source_percent,
                          images_by_topic, out_sync.proto_stamps_ns_by_topic,
                          nullptr)) {
        return false;
    }
    if (images_by_topic.size() != topics.size()) return false;

    std::vector<int64_t> ref_stamps;
    ref_stamps.reserve(out_sync.proto_stamps_ns_by_topic.size());
    for (const auto& [topic, stamp] : out_sync.proto_stamps_ns_by_topic) {
        (void)topic;
        ref_stamps.push_back(stamp);
    }
    std::sort(ref_stamps.begin(), ref_stamps.end());
    out_sync.ref_proto_ns = ref_stamps[ref_stamps.size() / 2];
    return true;
}

bool collectGroupSync(RosBagReader& reader,
                      const std::vector<std::string>& topics,
                      ExportGroupKind kind,
                      ExportGroupSync& out_sync) {
    out_sync = ExportGroupSync{};
    out_sync.kind = kind;
    if (!reader.isOpen() || topics.empty()) return false;

    if (!findBestAlignedPercent(reader, topics, 0.0, 1.0,
                                out_sync.source_percent,
                                out_sync.ref_proto_ns)) {
        return false;
    }

    std::map<std::string, cv::Mat> images_by_topic;
    if (!getAlignedImages(reader, topics, out_sync.source_percent,
                          images_by_topic, out_sync.proto_stamps_ns_by_topic,
                          nullptr)) {
        return false;
    }
    if (images_by_topic.size() != topics.size()) return false;

    std::vector<int64_t> ref_stamps;
    ref_stamps.reserve(out_sync.proto_stamps_ns_by_topic.size());
    for (const auto& [topic, stamp] : out_sync.proto_stamps_ns_by_topic) {
        (void)topic;
        ref_stamps.push_back(stamp);
    }
    std::sort(ref_stamps.begin(), ref_stamps.end());
    out_sync.ref_proto_ns = ref_stamps[ref_stamps.size() / 2];
    return true;
}

void collectCameraFramesAroundSync(RosBagReader& reader,
                                   const std::vector<std::string>& topics,
                                   const ExportGroupSync& group_sync,
                                   int64_t frame_half_ns,
                                   bool sync_frames_only,
                                   std::vector<SelectedBagMessage>& selected,
                                   std::set<std::string>& selected_keys) {
    if (topics.empty()) return;

    const int64_t window_lo = group_sync.ref_proto_ns - frame_half_ns;
    const int64_t window_hi = group_sync.ref_proto_ns + frame_half_ns;
    std::set<std::string> topic_set(topics.begin(), topics.end());

    reader.readMessages([&](const BagMessage& msg) {
        if (topic_set.count(msg.topic) == 0) return true;

        double ts_sec = 0;
        if (!ProtoDecoder::extractCompressedImageTimestampFromBag(msg.data,
                                                                  ts_sec)) {
            return true;
        }
        const int64_t proto_ns = static_cast<int64_t>(ts_sec * 1e9);

        auto ref_it = group_sync.proto_stamps_ns_by_topic.find(msg.topic);
        const bool is_sync_frame =
                ref_it != group_sync.proto_stamps_ns_by_topic.end() &&
                timeDiffNs(proto_ns, ref_it->second) < kImageSyncThresholdNs;
        if (sync_frames_only) {
            if (!is_sync_frame) return true;
        } else if (!is_sync_frame &&
                   !protoInWindow(proto_ns, window_lo, window_hi)) {
            return true;
        }

        const std::string key =
                msg.topic + ":" + std::to_string(msg.timestamp_ns);
        if (selected_keys.count(key) > 0) return true;
        selected_keys.insert(key);

        SelectedBagMessage entry;
        entry.msg = msg;
        entry.proto_ns = proto_ns;
        entry.group_ref_proto_ns =
                (ref_it != group_sync.proto_stamps_ns_by_topic.end())
                        ? ref_it->second
                        : group_sync.ref_proto_ns;
        entry.has_proto = true;
        selected.push_back(std::move(entry));
        return true;
    });
}

void collectCloudFramesNearRef(RosBagReader& reader,
                               const std::vector<std::string>& cloud_topics,
                               int64_t ref_proto_ns,
                               int64_t group_ref_proto_ns,
                               int64_t frame_half_ns,
                               std::vector<SelectedBagMessage>& selected,
                               std::set<std::string>& selected_keys) {
    if (ref_proto_ns <= 0 || group_ref_proto_ns <= 0) return;

    const auto active_topics = selectCloudTopics(reader, cloud_topics);
    if (active_topics.empty()) return;

    const int64_t window_lo = ref_proto_ns - frame_half_ns;
    const int64_t window_hi = ref_proto_ns + frame_half_ns;
    std::set<std::string> topic_set(active_topics.begin(), active_topics.end());

    reader.readMessages([&](const BagMessage& msg) {
        if (topic_set.count(msg.topic) == 0) return true;

        double ts_sec = 0;
        if (!ProtoDecoder::extractPointCloudTimestampFromBag(msg.data,
                                                             ts_sec)) {
            return true;
        }
        const int64_t proto_ns = static_cast<int64_t>(ts_sec * 1e9);
        if (!protoInWindow(proto_ns, window_lo, window_hi)) return true;

        const std::string key =
                msg.topic + ":" + std::to_string(msg.timestamp_ns);
        if (selected_keys.count(key) > 0) return true;
        selected_keys.insert(key);

        SelectedBagMessage entry;
        entry.msg = msg;
        entry.proto_ns = proto_ns;
        entry.group_ref_proto_ns = group_ref_proto_ns;
        entry.has_proto = true;
        selected.push_back(std::move(entry));
        return true;
    });
}

}  // namespace

bool exportMergedAlignedRosBag(const std::string& input_bag,
                               const std::string& output_bag,
                               const std::vector<std::string>& svm_image_topics,
                               const std::vector<std::string>& avm_image_topics,
                               const std::vector<std::string>& cloud_topics,
                               double output_duration_sec,
                               double frame_window_sec) {
    MergedBagExportOptions options;
    options.num_sync_groups = 1;
    options.output_duration_sec = output_duration_sec;
    options.frame_window_sec = frame_window_sec;
    return exportMergedAlignedRosBag(input_bag, output_bag, svm_image_topics,
                                     avm_image_topics, cloud_topics, options);
}

bool exportMergedAlignedRosBag(const std::string& input_bag,
                               const std::string& output_bag,
                               const std::vector<std::string>& svm_image_topics,
                               const std::vector<std::string>& avm_image_topics,
                               const std::vector<std::string>& cloud_topics,
                               const MergedBagExportOptions& options) {
    if (options.output_duration_sec <= 0.0) return false;

    RosBagReader reader;
    if (!reader.open(input_bag)) return false;

    const int num_groups = std::max(1, options.num_sync_groups);
    std::vector<double> centers = options.source_centers;
    if (centers.size() != static_cast<size_t>(num_groups)) {
        centers.resize(static_cast<size_t>(num_groups));
        for (int i = 0; i < num_groups; ++i) {
            centers[static_cast<size_t>(i)] =
                    (num_groups == 1)
                            ? 0.5
                            : (0.15 +
                               0.70 * static_cast<double>(i) /
                                       static_cast<double>(num_groups - 1));
        }
    }

    const int64_t frame_half_ns =
            options.sync_frames_only
                    ? kImageSyncThresholdNs
                    : static_cast<int64_t>(options.frame_window_sec * 1e9);
    const int64_t cloud_half_ns =
            options.sync_frames_only ? kCloudSyncThresholdNs : frame_half_ns;

    std::vector<SelectedBagMessage> selected;
    std::set<std::string> selected_keys;
    std::vector<ExportGroupSync> svm_syncs;
    std::vector<ExportGroupSync> avm_syncs;
    svm_syncs.reserve(static_cast<size_t>(num_groups));
    avm_syncs.reserve(static_cast<size_t>(num_groups));

    constexpr double kSearchHalf = 0.12;
    for (int gi = 0; gi < num_groups; ++gi) {
        ExportGroupSync svm_sync;
        ExportGroupSync avm_sync;
        if (!collectGroupSyncNear(
                    reader, svm_image_topics, ExportGroupKind::SVM,
                    centers[static_cast<size_t>(gi)], kSearchHalf, svm_sync)) {
            CVLog::Error(
                    "[BagAlignment] exportMergedAlignedRosBag: SVM sync failed "
                    "(group=%d center=%.2f)",
                    gi, centers[static_cast<size_t>(gi)]);
            return false;
        }
        if (!collectGroupSyncNear(
                    reader, avm_image_topics, ExportGroupKind::AVM,
                    centers[static_cast<size_t>(gi)], kSearchHalf, avm_sync)) {
            CVLog::Error(
                    "[BagAlignment] exportMergedAlignedRosBag: AVM sync failed "
                    "(group=%d center=%.2f)",
                    gi, centers[static_cast<size_t>(gi)]);
            return false;
        }
        svm_syncs.push_back(svm_sync);
        avm_syncs.push_back(avm_sync);

        collectCameraFramesAroundSync(reader, svm_image_topics, svm_sync,
                                      frame_half_ns, options.sync_frames_only,
                                      selected, selected_keys);
        collectCameraFramesAroundSync(reader, avm_image_topics, avm_sync,
                                      frame_half_ns, options.sync_frames_only,
                                      selected, selected_keys);
        collectCloudFramesNearRef(reader, cloud_topics, svm_sync.ref_proto_ns,
                                  svm_sync.ref_proto_ns, cloud_half_ns,
                                  selected, selected_keys);
    }

    size_t camera_count = 0;
    size_t cloud_count = 0;
    for (const auto& entry : selected) {
        if (entry.msg.topic.find("combined_point_cloud") != std::string::npos) {
            ++cloud_count;
        } else if (entry.msg.topic.find("/sensors/camera/") !=
                   std::string::npos) {
            ++camera_count;
        }
    }
    if (camera_count == 0) {
        CVLog::Warning(
                "[BagAlignment] exportMergedAlignedRosBag: no camera frames");
        return false;
    }

    uint64_t min_bag_ts = UINT64_MAX;
    uint64_t max_bag_ts = 0;
    for (const auto& entry : selected) {
        min_bag_ts = std::min(min_bag_ts, entry.msg.timestamp_ns);
        max_bag_ts = std::max(max_bag_ts, entry.msg.timestamp_ns);
    }
    const uint64_t bag_span =
            (max_bag_ts > min_bag_ts) ? (max_bag_ts - min_bag_ts) : 0;

    if (options.include_ancillary) {
        std::set<std::string> selected_topics;
        for (const auto& entry : selected) {
            selected_topics.insert(entry.msg.topic);
        }

        reader.readMessages([&](const BagMessage& msg) {
            if (selected_topics.count(msg.topic) > 0) return true;
            if (msg.timestamp_ns < min_bag_ts ||
                msg.timestamp_ns > max_bag_ts) {
                return true;
            }
            const std::string key =
                    msg.topic + ":" + std::to_string(msg.timestamp_ns);
            if (selected_keys.count(key) > 0) return true;
            selected_keys.insert(key);

            SelectedBagMessage entry;
            entry.msg = msg;
            entry.has_proto = false;
            selected.push_back(std::move(entry));
            return true;
        });
    }

    const uint64_t out_duration =
            static_cast<uint64_t>(options.output_duration_sec * 1e9);
    const uint64_t out_base = reader.getBeginTime();
    const uint64_t out_end = out_base + out_duration;

    auto output_center_for_group = [&](int group_index) -> uint64_t {
        if (num_groups <= 1) {
            return out_base + out_duration / 2;
        }
        const double frac = static_cast<double>(group_index) /
                            static_cast<double>(num_groups - 1);
        return out_base + static_cast<uint64_t>(frac * out_duration);
    };

    RosBagWriter writer;
    if (!writer.open(output_bag)) return false;

    std::map<std::string, uint32_t> topic_conn;
    for (const auto& [id, conn] : reader.getConnections()) {
        (void)id;
        topic_conn[conn.topic] = writer.addConnection(conn);
    }

    size_t written = 0;
    for (const auto& entry : selected) {
        auto it = topic_conn.find(entry.msg.topic);
        if (it == topic_conn.end()) continue;

        uint64_t out_ts = out_base + out_duration / 2;
        if (entry.has_proto) {
            int group_idx = 0;
            for (int gi = 0; gi < num_groups; ++gi) {
                if (timeDiffNs(
                            entry.group_ref_proto_ns,
                            svm_syncs[static_cast<size_t>(gi)].ref_proto_ns) <
                    kImageSyncThresholdNs) {
                    group_idx = gi;
                    break;
                }
                if (timeDiffNs(
                            entry.group_ref_proto_ns,
                            avm_syncs[static_cast<size_t>(gi)].ref_proto_ns) <
                    kImageSyncThresholdNs) {
                    group_idx = gi;
                    break;
                }
            }
            const uint64_t out_center = output_center_for_group(group_idx);
            out_ts = remapProtoNearCenter(entry.proto_ns,
                                          entry.group_ref_proto_ns, out_center);
            if (out_ts < out_base) out_ts = out_base;
            if (out_ts > out_end) out_ts = out_end;
        } else if (options.include_ancillary) {
            out_ts = mapBagToOutputTime(entry.msg.timestamp_ns, min_bag_ts,
                                        bag_span, out_base, out_duration);
        } else {
            continue;
        }
        writer.writeMessage(it->second, out_ts, entry.msg.data);
        ++written;
    }

    if (!writer.close()) return false;

    const int64_t svm_avm_gap_ns =
            (num_groups > 0) ? timeDiffNs(svm_syncs.front().ref_proto_ns,
                                          avm_syncs.front().ref_proto_ns)
                             : 0;
    CVLog::Print(
            "[BagAlignment] exportMergedAlignedRosBag: %s -> %s "
            "(%zu msgs, groups=%d, %.3fs, sync_only=%d ancillary=%d "
            "camera=%zu cloud=%zu svm0=%.3f avm0=%.3f gap0=%.3fs)",
            input_bag.c_str(), output_bag.c_str(), written, num_groups,
            options.output_duration_sec, options.sync_frames_only ? 1 : 0,
            options.include_ancillary ? 1 : 0, camera_count, cloud_count,
            svm_syncs.empty() ? 0.0 : svm_syncs.front().source_percent,
            avm_syncs.empty() ? 0.0 : avm_syncs.front().source_percent,
            static_cast<double>(svm_avm_gap_ns) / 1e9);
    return written > 0;
}

bool exportBevAlignedRosBag(const std::string& input_bag,
                            const std::string& output_bag,
                            const std::vector<std::string>& svm_image_topics,
                            const std::vector<std::string>& avm_image_topics,
                            double sync_percent,
                            double proto_window_sec) {
    (void)sync_percent;
    return exportMergedAlignedRosBag(
            input_bag, output_bag, svm_image_topics, avm_image_topics,
            {"/sensors/lidar/combined_point_cloud_proto"}, proto_window_sec,
            proto_window_sec * 0.5);
}

bool isSplitTimelineBevBag(RosBagReader& reader,
                           const std::vector<std::string>& svm_image_topics,
                           const std::vector<std::string>& avm_image_topics) {
    if (!reader.isOpen() || svm_image_topics.empty() ||
        avm_image_topics.empty()) {
        return false;
    }

    auto median_bag_ts =
            [&](const std::vector<std::string>& topics) -> uint64_t {
        std::vector<uint64_t> stamps;
        stamps.reserve(topics.size());
        for (const auto& topic : topics) {
            const auto msg = reader.readMessageAtPercent(topic, 0.5);
            if (!msg.data.empty()) {
                stamps.push_back(msg.timestamp_ns);
            }
        }
        if (stamps.empty()) return 0;
        std::sort(stamps.begin(), stamps.end());
        return stamps[stamps.size() / 2];
    };

    const uint64_t svm_med = median_bag_ts(svm_image_topics);
    const uint64_t avm_med = median_bag_ts(avm_image_topics);
    const uint64_t duration = reader.getDuration();
    if (svm_med == 0 || avm_med == 0 || duration == 0) return false;
    return avm_med > svm_med + duration / 3;
}

static double groupPlaybackPercent(double percent,
                                   bool is_avm_group,
                                   bool split_timeline) {
    if (!split_timeline) return percent;
    if (is_avm_group) return 0.5 + percent * 0.5;
    return percent * 0.5;
}

bool getAlignedImagesBevGroups(
        RosBagReader& reader,
        const std::vector<std::string>& svm_image_topics,
        const std::vector<std::string>& avm_image_topics,
        double percent,
        std::map<std::string, cv::Mat>& images_by_topic,
        std::map<std::string, int64_t>& stamps_ns_by_topic,
        VehicleStateData* vehicle_state_out) {
    images_by_topic.clear();
    stamps_ns_by_topic.clear();
    if (vehicle_state_out) {
        *vehicle_state_out = VehicleStateData{};
    }
    if (!reader.isOpen()) return false;

    const bool split_timeline =
            isSplitTimelineBevBag(reader, svm_image_topics, avm_image_topics);

    auto merge_group = [&](const std::vector<std::string>& topics,
                           bool is_avm_group) -> bool {
        if (topics.empty()) return false;
        // manual_sensor_calib: one percent drives bag-record search window;
        // frame match uses proto timestamps within 25ms.
        // Remapped merge bags may still remap AVM to the upper bag half.
        const double group_percent =
                split_timeline
                        ? groupPlaybackPercent(percent, is_avm_group, true)
                        : percent;
        std::map<std::string, cv::Mat> partial_images;
        std::map<std::string, int64_t> partial_stamps;
        if (!getAlignedImages(reader, topics, group_percent, partial_images,
                              partial_stamps, vehicle_state_out)) {
            CVLog::Warning(
                    "[BagAlignment] BEV group sync failed: %zu topics at "
                    "%.1f%% "
                    "(user=%.1f%%, split=%d)",
                    topics.size(), group_percent * 100.0, percent * 100.0,
                    split_timeline ? 1 : 0);
            return false;
        }
        for (auto& [topic, image] : partial_images) {
            images_by_topic[topic] = std::move(image);
        }
        for (const auto& [topic, stamp] : partial_stamps) {
            stamps_ns_by_topic[topic] = stamp;
        }
        CVLog::Print("[BagAlignment] BEV group synced: %zu/%zu topics",
                     partial_images.size(), topics.size());
        return true;
    };

    const bool svm_ok = svm_image_topics.empty()
                                ? true
                                : merge_group(svm_image_topics, false);
    const bool avm_ok = avm_image_topics.empty()
                                ? true
                                : merge_group(avm_image_topics, true);
    if (!svm_ok || !avm_ok) {
        images_by_topic.clear();
        stamps_ns_by_topic.clear();
        return false;
    }
    const size_t expected = svm_image_topics.size() + avm_image_topics.size();
    if (expected > 0 && images_by_topic.size() != expected) {
        CVLog::Warning(
                "[BagAlignment] BEV group sync incomplete: images=%zu/%zu",
                images_by_topic.size(), expected);
        images_by_topic.clear();
        stamps_ns_by_topic.clear();
        return false;
    }
    return !images_by_topic.empty();
}

}  // namespace mcalib
