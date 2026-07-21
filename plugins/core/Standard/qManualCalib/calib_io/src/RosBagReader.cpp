// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RosBagReader.h"

#include <CPUInfo.h>
#include <CVLog.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

#include "ProtoDecoder.h"
#include "VideoCodecDecoder.h"
#include "mcalib_portability.h"

#ifdef MCALIB_HAS_BZ2
#include <bzlib.h>
#endif

#ifdef MCALIB_HAS_LZ4
#include <lz4.h>
#endif

namespace mcalib {

static constexpr const char* ROSBAG_MAGIC = "#ROSBAG V2.0\n";
static constexpr int ROSBAG_MAGIC_LEN = 13;

enum RecordType : uint8_t {
    MSGDATA = 0x02,
    BAG_HEADER = 0x03,
    INDEX_DATA = 0x04,
    CHUNK = 0x05,
    CHUNK_INFO = 0x06,
    CONNECTION = 0x07
};

std::string BagMessage::getStringPayload() const {
    if (data.size() < 4) return {};
    uint32_t len = RosBagReader::readUint32(data.data());
    if (4 + len > data.size()) return {};
    return data.substr(4, len);
}

RosBagReader::RosBagReader() = default;
RosBagReader::~RosBagReader() { close(); }

void RosBagReader::resetMultiState() {
    sub_readers_.clear();
    topic_owner_idx_.clear();
    source_bag_paths_.clear();
    multi_mode_ = false;
}

RosBagReader* RosBagReader::readerForTopic(const std::string& topic) {
    if (!multi_mode_) return this;
    auto it = topic_owner_idx_.find(topic);
    if (it == topic_owner_idx_.end()) return nullptr;
    return sub_readers_[it->second].reader.get();
}

const RosBagReader* RosBagReader::readerForTopic(
        const std::string& topic) const {
    if (!multi_mode_) return this;
    auto it = topic_owner_idx_.find(topic);
    if (it == topic_owner_idx_.end()) return nullptr;
    return sub_readers_[it->second].reader.get();
}

bool RosBagReader::openMulti(const std::vector<std::string>& paths) {
    close();
    if (paths.empty()) return false;
    if (paths.size() == 1) return open(paths.front());

    multi_mode_ = true;
    source_bag_paths_ = paths;
    path_ = paths.front();
    bag_begin_time_ = UINT64_MAX;
    bag_end_time_ = 0;
    total_message_count_ = 0;

    for (const auto& bag_path : paths) {
        auto reader = std::make_unique<RosBagReader>();
        if (!reader->open(bag_path)) {
            CVLog::Warning("[RosBagReader] openMulti skip unreadable bag: %s",
                           bag_path.c_str());
            continue;
        }

        bag_begin_time_ = std::min(bag_begin_time_, reader->getBeginTime());
        bag_end_time_ = std::max(bag_end_time_, reader->getEndTime());
        total_message_count_ += reader->getMessageCount();

        for (const auto& topic : reader->getTopics()) {
            if (topic_owner_idx_.count(topic)) {
                CVLog::Warning(
                        "[RosBagReader] openMulti duplicate topic '%s', "
                        "keeping first bag",
                        topic.c_str());
                continue;
            }
            topic_owner_idx_[topic] = sub_readers_.size();
            const auto types = reader->getTopicTypes();
            auto type_it = types.find(topic);
            if (type_it != types.end()) {
                const uint32_t conn_id =
                        static_cast<uint32_t>(topic_to_conn_.size());
                BagConnection conn;
                conn.id = conn_id;
                conn.topic = topic;
                conn.datatype = type_it->second;
                connections_[conn_id] = conn;
                topic_to_conn_[topic] = conn_id;
            }
        }

        SubBagReader entry;
        entry.path = bag_path;
        entry.reader = std::move(reader);
        sub_readers_.push_back(std::move(entry));
    }

    is_open_ = !sub_readers_.empty();
    if (!is_open_) {
        resetMultiState();
        return false;
    }

    CVLog::Print(
            "[RosBagReader] openMulti: %zu bags, %zu topics, duration=%.1fs",
            sub_readers_.size(), topic_owner_idx_.size(), getDuration() / 1e9);
    return true;
}

bool RosBagReader::open(const std::string& path) {
    close();
    path_ = path;
    if (!openInputFile(file_, path)) {
        CVLog::Error("[RosBagReader] Failed to open: %s", path.c_str());
        return false;
    }

    if (!readVersion()) {
        CVLog::Error(
                "[RosBagReader] Invalid bag file or unsupported version: %s",
                path.c_str());
        close();
        return false;
    }

    if (!readBagHeader()) {
        CVLog::Error("[RosBagReader] Failed to read bag header");
        close();
        return false;
    }

    if (!readRecords()) {
        CVLog::Error("[RosBagReader] Failed to read records");
        close();
        return false;
    }

    is_open_ = true;

    CVLog::Print(
            "[RosBagReader] Opened: %s (%u chunks, %u connections, "
            "duration=%.1fs)",
            path.c_str(), static_cast<unsigned>(chunk_records_.size()),
            static_cast<unsigned>(connections_.size()), getDuration() / 1e9);

    return true;
}

void RosBagReader::close() {
    if (file_.is_open()) file_.close();
    clearVideoDecodeCache();
    resetMultiState();
    is_open_ = false;
    connections_.clear();
    topic_to_conn_.clear();
    chunk_infos_.clear();
    chunk_records_.clear();
    bag_begin_time_ = UINT64_MAX;
    bag_end_time_ = 0;
    total_message_count_ = 0;
    clearTopicTimeIndex();
}

void RosBagReader::clearTopicTimeIndex() {
    if (multi_mode_) {
        for (auto& sub : sub_readers_) {
            sub.reader->clearTopicTimeIndex();
        }
    }
    {
        std::lock_guard<std::mutex> lock(index_mutex_);
        topic_time_index_.clear();
        index_built_ = false;
    }
    index_ready_.store(false, std::memory_order_release);

    if (!multi_mode_) {
        std::lock_guard<std::mutex> lock(chunk_cache_mutex_);
        chunk_cache_.clear();
        chunk_cache_lru_.clear();
    }
}

bool RosBagReader::isOpen() const { return is_open_; }

uint64_t RosBagReader::getBeginTime() const { return bag_begin_time_; }
uint64_t RosBagReader::getEndTime() const { return bag_end_time_; }
uint64_t RosBagReader::getDuration() const {
    return bag_end_time_ > bag_begin_time_ ? bag_end_time_ - bag_begin_time_
                                           : 0;
}

VideoDecodeCache& RosBagReader::videoDecodeCache() {
    if (!video_decode_cache_) {
        video_decode_cache_ = std::make_unique<VideoDecodeCache>();
    }
    return *video_decode_cache_;
}

void RosBagReader::clearVideoDecodeCache() {
    if (video_decode_cache_) {
        video_decode_cache_->clear();
        video_decode_cache_.reset();
    }
}

namespace {

bool refinePlaybackRangeFromBagStamps(RosBagReader& reader,
                                      const std::string& topic,
                                      uint64_t& out_begin,
                                      uint64_t& out_end) {
    uint64_t first_stamp = 0;
    uint64_t last_stamp = 0;
    reader.readMessages(
            [&](const BagMessage& msg) {
                if (msg.timestamp_ns == 0) return true;
                if (first_stamp == 0 || msg.timestamp_ns < first_stamp) {
                    first_stamp = msg.timestamp_ns;
                }
                if (msg.timestamp_ns > last_stamp) {
                    last_stamp = msg.timestamp_ns;
                }
                return true;
            },
            {topic});
    if (first_stamp == 0 || last_stamp == 0 || last_stamp <= first_stamp) {
        return false;
    }
    out_begin = first_stamp;
    out_end = last_stamp;
    return true;
}

}  // namespace

bool RosBagReader::refinePlaybackTimeRange(const std::string& topic) {
    if (!is_open_) return false;
    if (multi_mode_) {
        RosBagReader* owner = readerForTopic(topic);
        if (!owner) {
            CVLog::Warning(
                    "[RosBagReader] refinePlaybackTimeRange topic '%s' not "
                    "found in multi-bag set",
                    topic.c_str());
            return false;
        }
        uint64_t first_stamp = 0;
        uint64_t last_stamp = 0;
        if (!refinePlaybackRangeFromBagStamps(*owner, topic, first_stamp,
                                              last_stamp)) {
            CVLog::Warning(
                    "[RosBagReader] refinePlaybackTimeRange failed for '%s', "
                    "using bag header time",
                    topic.c_str());
            return false;
        }
        const uint64_t orig_begin = bag_begin_time_;
        const uint64_t orig_end = bag_end_time_;
        bag_begin_time_ = first_stamp;
        bag_end_time_ = last_stamp;
        CVLog::Print(
                "[RosBagReader] refinePlaybackTimeRange '%s': [%llu,%llu] -> "
                "[%llu,%llu] ns (bag stamps, multi)",
                topic.c_str(), static_cast<unsigned long long>(orig_begin),
                static_cast<unsigned long long>(orig_end),
                static_cast<unsigned long long>(first_stamp),
                static_cast<unsigned long long>(last_stamp));
        return true;
    }

    const uint64_t orig_begin = bag_begin_time_;
    const uint64_t orig_end = bag_end_time_;
    uint64_t first_stamp = 0;
    uint64_t last_stamp = 0;
    if (!refinePlaybackRangeFromBagStamps(*this, topic, first_stamp,
                                          last_stamp)) {
        CVLog::Warning(
                "[RosBagReader] refinePlaybackTimeRange failed for '%s', "
                "using bag header time",
                topic.c_str());
        return false;
    }

    CVLog::Print(
            "[RosBagReader] refinePlaybackTimeRange '%s': [%llu,%llu] -> "
            "[%llu,%llu] ns (bag stamps)",
            topic.c_str(), static_cast<unsigned long long>(orig_begin),
            static_cast<unsigned long long>(orig_end),
            static_cast<unsigned long long>(first_stamp),
            static_cast<unsigned long long>(last_stamp));

    bag_begin_time_ = first_stamp;
    bag_end_time_ = last_stamp;
    return true;
}

uint32_t RosBagReader::getMessageCount() const { return total_message_count_; }

std::vector<std::string> RosBagReader::getTopics() const {
    if (multi_mode_) {
        std::vector<std::string> topics;
        topics.reserve(topic_owner_idx_.size());
        for (const auto& [topic, _] : topic_owner_idx_) {
            topics.push_back(topic);
        }
        std::sort(topics.begin(), topics.end());
        return topics;
    }
    std::vector<std::string> topics;
    topics.reserve(topic_to_conn_.size());
    for (const auto& [topic, _] : topic_to_conn_) {
        topics.push_back(topic);
    }
    return topics;
}

std::map<std::string, std::string> RosBagReader::getTopicTypes() const {
    if (multi_mode_) {
        std::map<std::string, std::string> result;
        for (const auto& sub : sub_readers_) {
            const auto types = sub.reader->getTopicTypes();
            for (const auto& [topic, type] : types) {
                result.emplace(topic, type);
            }
        }
        return result;
    }
    std::map<std::string, std::string> result;
    for (const auto& [id, conn] : connections_) {
        result[conn.topic] = conn.datatype;
    }
    return result;
}

uint32_t RosBagReader::readUint32(const char* data) {
    uint32_t val;
    std::memcpy(&val, data, 4);
    return val;
}

uint64_t RosBagReader::readUint64(const char* data) {
    uint64_t val;
    std::memcpy(&val, data, 8);
    return val;
}

std::map<std::string, std::string> RosBagReader::parseHeader(const char* data,
                                                             uint32_t len) {
    std::map<std::string, std::string> fields;
    uint32_t pos = 0;
    while (pos < len) {
        if (pos + 4 > len) break;
        uint32_t field_len = readUint32(data + pos);
        pos += 4;
        if (pos + field_len > len) break;

        std::string field(data + pos, field_len);
        pos += field_len;

        auto eq = field.find('=');
        if (eq != std::string::npos) {
            std::string key = field.substr(0, eq);
            std::string val = field.substr(eq + 1);
            fields[key] = val;
        }
    }
    return fields;
}

bool RosBagReader::readVersion() {
    char magic[ROSBAG_MAGIC_LEN];
    file_.read(magic, ROSBAG_MAGIC_LEN);
    if (!file_.good()) return false;
    return std::memcmp(magic, ROSBAG_MAGIC, ROSBAG_MAGIC_LEN) == 0;
}

bool RosBagReader::readBagHeader() {
    uint32_t header_len;
    file_.read(reinterpret_cast<char*>(&header_len), 4);
    if (!file_.good()) return false;

    std::vector<char> header_data(header_len);
    file_.read(header_data.data(), header_len);
    if (!file_.good()) return false;

    auto fields = parseHeader(header_data.data(), header_len);

    if (fields.find("op") == fields.end()) return false;
    uint8_t op = static_cast<uint8_t>(fields["op"][0]);
    if (op != BAG_HEADER) return false;

    if (fields.count("index_pos") && fields["index_pos"].size() >= 8) {
        index_pos_ = readUint64(fields["index_pos"].data());
    }
    if (fields.count("chunk_count") && fields["chunk_count"].size() >= 4) {
        chunk_count_ = readUint32(fields["chunk_count"].data());
    }
    if (fields.count("conn_count") && fields["conn_count"].size() >= 4) {
        conn_count_ = readUint32(fields["conn_count"].data());
    }

    uint32_t data_len;
    file_.read(reinterpret_cast<char*>(&data_len), 4);
    if (!file_.good()) return false;
    file_.seekg(data_len, std::ios::cur);

    return true;
}

bool RosBagReader::readRecords() {
    while (file_.good() && !file_.eof()) {
        uint32_t header_len;
        file_.read(reinterpret_cast<char*>(&header_len), 4);
        if (!file_.good() || file_.eof()) break;

        if (header_len > 64 * 1024 * 1024) {
            CVLog::Warning("[RosBagReader] Suspiciously large header: %u bytes",
                           header_len);
            break;
        }

        std::vector<char> header_data(header_len);
        file_.read(header_data.data(), header_len);
        if (!file_.good()) break;

        uint32_t data_len;
        file_.read(reinterpret_cast<char*>(&data_len), 4);
        if (!file_.good()) break;

        auto fields = parseHeader(header_data.data(), header_len);
        if (fields.find("op") == fields.end()) {
            file_.seekg(data_len, std::ios::cur);
            continue;
        }

        uint8_t op = static_cast<uint8_t>(fields["op"][0]);

        switch (op) {
            case CHUNK: {
                ChunkRecord cr;
                if (fields.count("compression")) {
                    cr.compression = fields["compression"];
                }
                if (fields.count("size") && fields["size"].size() >= 4) {
                    cr.uncompressed_size = readUint32(fields["size"].data());
                }
                cr.compressed_size = data_len;
                cr.data_pos = static_cast<uint64_t>(file_.tellg());
                chunk_records_.push_back(cr);
                file_.seekg(data_len, std::ios::cur);
                break;
            }
            case CONNECTION: {
                BagConnection conn;
                if (fields.count("conn") && fields["conn"].size() >= 4) {
                    conn.id = readUint32(fields["conn"].data());
                }
                if (fields.count("topic")) {
                    conn.topic = fields["topic"];
                }

                std::vector<char> conn_data(data_len);
                file_.read(conn_data.data(), data_len);
                auto conn_fields = parseHeader(conn_data.data(), data_len);

                if (conn_fields.count("type"))
                    conn.datatype = conn_fields["type"];
                if (conn_fields.count("md5sum"))
                    conn.md5sum = conn_fields["md5sum"];
                if (conn_fields.count("message_definition")) {
                    conn.message_definition = conn_fields["message_definition"];
                }
                if (conn_fields.count("topic") && conn.topic.empty()) {
                    conn.topic = conn_fields["topic"];
                }

                connections_[conn.id] = conn;
                topic_to_conn_[conn.topic] = conn.id;
                break;
            }
            case CHUNK_INFO: {
                std::vector<char> ci_data(data_len);
                file_.read(ci_data.data(), data_len);

                ChunkInfo ci;
                if (fields.count("chunk_pos") &&
                    fields["chunk_pos"].size() >= 8) {
                    ci.chunk_pos = readUint64(fields["chunk_pos"].data());
                }
                if (fields.count("start_time") &&
                    fields["start_time"].size() >= 8) {
                    uint32_t secs = readUint32(fields["start_time"].data());
                    uint32_t nsecs =
                            readUint32(fields["start_time"].data() + 4);
                    ci.start_time =
                            static_cast<uint64_t>(secs) * 1000000000ULL + nsecs;
                }
                if (fields.count("end_time") &&
                    fields["end_time"].size() >= 8) {
                    uint32_t secs = readUint32(fields["end_time"].data());
                    uint32_t nsecs = readUint32(fields["end_time"].data() + 4);
                    ci.end_time =
                            static_cast<uint64_t>(secs) * 1000000000ULL + nsecs;
                }
                if (fields.count("count") && fields["count"].size() >= 4) {
                    ci.count = readUint32(fields["count"].data());
                }

                chunk_infos_.push_back(ci);
                total_message_count_ += ci.count;

                if (ci.start_time < bag_begin_time_)
                    bag_begin_time_ = ci.start_time;
                if (ci.end_time > bag_end_time_) bag_end_time_ = ci.end_time;
                break;
            }
            case INDEX_DATA:
            case MSGDATA:
            default:
                file_.seekg(data_len, std::ios::cur);
                break;
        }
    }

    if (bag_begin_time_ == UINT64_MAX) bag_begin_time_ = 0;

    CVLog::Print(
            "[RosBagReader] readRecords: %zu connections, %zu chunks, "
            "%zu chunk_infos, time=[%lu, %lu]ns, msgs=%u",
            connections_.size(), chunk_records_.size(), chunk_infos_.size(),
            static_cast<unsigned long>(bag_begin_time_),
            static_cast<unsigned long>(bag_end_time_), total_message_count_);
    return true;
}

bool RosBagReader::parseChunkMessages(const char* chunk_data,
                                      uint32_t chunk_size,
                                      const std::set<uint32_t>& conn_ids,
                                      uint64_t start_time_ns,
                                      uint64_t end_time_ns,
                                      const MessageCallback& callback) {
    uint32_t pos = 0;
    while (pos < chunk_size) {
        if (pos + 4 > chunk_size) break;
        uint32_t header_len = readUint32(chunk_data + pos);
        pos += 4;
        if (pos + header_len > chunk_size) break;

        auto fields = parseHeader(chunk_data + pos, header_len);
        pos += header_len;

        if (pos + 4 > chunk_size) break;
        uint32_t data_len = readUint32(chunk_data + pos);
        pos += 4;
        if (pos + data_len > chunk_size) break;

        if (fields.find("op") == fields.end()) {
            pos += data_len;
            continue;
        }

        uint8_t op = static_cast<uint8_t>(fields["op"][0]);

        if (op == MSGDATA) {
            uint32_t conn_id = 0;
            uint64_t time_ns = 0;

            if (fields.count("conn") && fields["conn"].size() >= 4) {
                conn_id = readUint32(fields["conn"].data());
            }
            if (fields.count("time") && fields["time"].size() >= 8) {
                uint32_t secs = readUint32(fields["time"].data());
                uint32_t nsecs = readUint32(fields["time"].data() + 4);
                time_ns = static_cast<uint64_t>(secs) * 1000000000ULL + nsecs;
            }

            if (time_ns >= start_time_ns && time_ns <= end_time_ns) {
                if (conn_ids.empty() || conn_ids.count(conn_id)) {
                    BagMessage msg;
                    if (connections_.count(conn_id)) {
                        msg.topic = connections_[conn_id].topic;
                    }
                    msg.timestamp_ns = time_ns;
                    msg.data.assign(chunk_data + pos, data_len);

                    if (!callback(msg)) return false;
                }
            }
        }
        pos += data_len;
    }
    return true;
}

bool RosBagReader::readChunkRaw(const ChunkRecord& cr,
                                std::vector<char>& raw_data) const {
    file_.clear();
    file_.seekg(cr.data_pos);
    if (!file_.good()) return false;
    raw_data.resize(cr.compressed_size);
    file_.read(raw_data.data(), cr.compressed_size);
    return file_.good();
}

bool RosBagReader::decompressChunk(const ChunkRecord& cr,
                                   const std::vector<char>& raw_data,
                                   std::vector<char>& decompressed) const {
    if (cr.compression == "bz2") {
        return decompressBZ2(raw_data.data(), cr.compressed_size,
                             cr.uncompressed_size, decompressed);
    } else if (cr.compression == "lz4") {
        return decompressLZ4(raw_data.data(), cr.compressed_size,
                             cr.uncompressed_size, decompressed);
    } else if (cr.compression == "none" || cr.compression.empty()) {
        decompressed.assign(raw_data.begin(), raw_data.end());
        return true;
    }
    CVLog::Warning("[RosBagReader] Unsupported compression: %s",
                   cr.compression.c_str());
    return false;
}

bool RosBagReader::readChunkData(const ChunkRecord& cr,
                                 const std::set<uint32_t>& conn_ids,
                                 uint64_t start_time_ns,
                                 uint64_t end_time_ns,
                                 const MessageCallback& callback) {
    std::vector<char> raw_data;
    if (!readChunkRaw(cr, raw_data)) {
        CVLog::Warning("[RosBagReader] Failed to read chunk at pos %llu",
                       static_cast<unsigned long long>(cr.data_pos));
        return false;
    }

    std::vector<char> decompressed;
    if (!decompressChunk(cr, raw_data, decompressed)) {
        CVLog::Warning("[RosBagReader] Decompression failed at pos %llu",
                       static_cast<unsigned long long>(cr.data_pos));
        return false;
    }

    return parseChunkMessages(decompressed.data(),
                              static_cast<uint32_t>(decompressed.size()),
                              conn_ids, start_time_ns, end_time_ns, callback);
}

std::vector<BagMessage> RosBagReader::parseChunkMessagesToVec(
        const char* chunk_data,
        uint32_t chunk_size,
        const std::set<uint32_t>& conn_ids,
        uint64_t start_time_ns,
        uint64_t end_time_ns) {
    std::vector<BagMessage> result;

    uint32_t pos = 0;
    while (pos < chunk_size) {
        if (pos + 4 > chunk_size) break;
        uint32_t header_len = readUint32(chunk_data + pos);
        pos += 4;
        if (pos + header_len > chunk_size) break;

        auto fields = parseHeader(chunk_data + pos, header_len);
        pos += header_len;

        if (pos + 4 > chunk_size) break;
        uint32_t data_len = readUint32(chunk_data + pos);
        pos += 4;
        if (pos + data_len > chunk_size) break;

        if (fields.find("op") == fields.end()) {
            pos += data_len;
            continue;
        }

        uint8_t op = static_cast<uint8_t>(fields["op"][0]);

        if (op == MSGDATA) {
            uint32_t conn_id = 0;
            uint64_t time_ns = 0;

            if (fields.count("conn") && fields["conn"].size() >= 4) {
                conn_id = readUint32(fields["conn"].data());
            }
            if (fields.count("time") && fields["time"].size() >= 8) {
                uint32_t secs = readUint32(fields["time"].data());
                uint32_t nsecs = readUint32(fields["time"].data() + 4);
                time_ns = static_cast<uint64_t>(secs) * 1000000000ULL + nsecs;
            }

            if (time_ns >= start_time_ns && time_ns <= end_time_ns) {
                if (conn_ids.empty() || conn_ids.count(conn_id)) {
                    BagMessage msg;
                    if (connections_.count(conn_id)) {
                        msg.topic = connections_[conn_id].topic;
                    }
                    msg.timestamp_ns = time_ns;
                    msg.data.assign(chunk_data + pos, data_len);
                    result.push_back(std::move(msg));
                }
            }
        }
        pos += data_len;
    }
    return result;
}

std::vector<BagMessage> RosBagReader::readAllMessagesParallel(
        const std::set<std::string>& topics,
        std::function<void(int)> progress_cb,
        uint64_t start_time_ns,
        uint64_t end_time_ns,
        int num_threads) {
    if (!is_open_) return {};

    if (multi_mode_) {
        std::vector<BagMessage> all;
        for (size_t i = 0; i < sub_readers_.size(); ++i) {
            std::set<std::string> sub_topics;
            if (!topics.empty()) {
                for (const auto& topic : topics) {
                    auto owner_it = topic_owner_idx_.find(topic);
                    if (owner_it != topic_owner_idx_.end() &&
                        owner_it->second == i) {
                        sub_topics.insert(topic);
                    }
                }
                if (sub_topics.empty()) continue;
            }
            auto part = sub_readers_[i].reader->readAllMessagesParallel(
                    sub_topics, progress_cb, start_time_ns, end_time_ns,
                    num_threads);
            all.insert(all.end(), std::make_move_iterator(part.begin()),
                       std::make_move_iterator(part.end()));
        }
        std::sort(all.begin(), all.end(),
                  [](const BagMessage& a, const BagMessage& b) {
                      return a.timestamp_ns < b.timestamp_ns;
                  });
        return all;
    }

    std::set<uint32_t> conn_ids;
    if (!topics.empty()) {
        for (const auto& topic : topics) {
            auto it = topic_to_conn_.find(topic);
            if (it != topic_to_conn_.end()) {
                conn_ids.insert(it->second);
            }
        }
        if (conn_ids.empty()) return {};
    }

    struct ChunkWork {
        size_t chunk_idx;
        ChunkRecord record;
        std::vector<char> raw_data;
    };

    std::vector<ChunkWork> work_items;
    for (size_t i = 0; i < chunk_records_.size(); ++i) {
        if (i < chunk_infos_.size()) {
            const auto& ci = chunk_infos_[i];
            if (ci.end_time < start_time_ns || ci.start_time > end_time_ns)
                continue;
        }
        ChunkWork cw;
        cw.chunk_idx = i;
        cw.record = chunk_records_[i];
        if (!readChunkRaw(cw.record, cw.raw_data)) continue;
        work_items.push_back(std::move(cw));
    }

    std::vector<std::vector<BagMessage>> per_chunk_msgs(work_items.size());

    size_t total = work_items.size();
    std::atomic<size_t> done_count{0};
    auto tracked_worker = [&](size_t start, size_t end) {
        for (size_t wi = start; wi < end; ++wi) {
            auto& cw = work_items[wi];
            std::vector<char> decompressed;
            if (!decompressChunk(cw.record, cw.raw_data, decompressed)) {
                ++done_count;
                continue;
            }
            cw.raw_data.clear();
            cw.raw_data.shrink_to_fit();
            per_chunk_msgs[wi] = parseChunkMessagesToVec(
                    decompressed.data(),
                    static_cast<uint32_t>(decompressed.size()), conn_ids,
                    start_time_ns, end_time_ns);
            size_t d = ++done_count;
            if (progress_cb && total > 0) {
                progress_cb(static_cast<int>(d * 100 / total));
            }
        }
    };

    if (num_threads <= 0) {
        num_threads = cloudViewer::utility::CPUInfo::GetInstance().NumThreads();
        if (num_threads <= 0) num_threads = 4;
    }

    if (num_threads <= 1 || total <= 1) {
        tracked_worker(0, total);
    } else {
        size_t n = std::min(static_cast<size_t>(num_threads), total);
        std::vector<std::thread> threads;
        size_t per_thread = (total + n - 1) / n;
        for (size_t t = 0; t < n; ++t) {
            size_t s = t * per_thread;
            size_t e = std::min(s + per_thread, total);
            if (s >= total) break;
            threads.emplace_back(tracked_worker, s, e);
        }
        for (auto& th : threads) th.join();
    }

    std::vector<BagMessage> all;
    for (auto& msgs : per_chunk_msgs) {
        all.insert(all.end(), std::make_move_iterator(msgs.begin()),
                   std::make_move_iterator(msgs.end()));
    }
    std::sort(all.begin(), all.end(),
              [](const BagMessage& a, const BagMessage& b) {
                  return a.timestamp_ns < b.timestamp_ns;
              });
    return all;
}

bool RosBagReader::readMessages(const MessageCallback& callback,
                                const std::set<std::string>& topics,
                                uint64_t start_time_ns,
                                uint64_t end_time_ns) {
    if (!is_open_) {
        CVLog::Warning("[RosBagReader] Bag file is not open");
        return false;
    }

    if (multi_mode_) {
        bool ok = true;
        for (size_t i = 0; i < sub_readers_.size(); ++i) {
            std::set<std::string> sub_topics;
            if (!topics.empty()) {
                for (const auto& topic : topics) {
                    auto owner_it = topic_owner_idx_.find(topic);
                    if (owner_it != topic_owner_idx_.end() &&
                        owner_it->second == i) {
                        sub_topics.insert(topic);
                    }
                }
                if (sub_topics.empty()) continue;
            }
            ok = sub_readers_[i].reader->readMessages(
                         callback, sub_topics, start_time_ns, end_time_ns) &&
                 ok;
        }
        return ok;
    }

    std::set<uint32_t> conn_ids;
    if (!topics.empty()) {
        for (const auto& topic : topics) {
            auto it = topic_to_conn_.find(topic);
            if (it != topic_to_conn_.end()) {
                conn_ids.insert(it->second);
            }
        }
        if (conn_ids.empty()) return true;
    }

    for (size_t i = 0; i < chunk_records_.size(); ++i) {
        if (i < chunk_infos_.size()) {
            const auto& ci = chunk_infos_[i];
            if (ci.end_time < start_time_ns || ci.start_time > end_time_ns) {
                continue;
            }
        }
        if (!readChunkData(chunk_records_[i], conn_ids, start_time_ns,
                           end_time_ns, callback)) {
            return false;
        }
    }
    return true;
}

std::vector<BagMessage> RosBagReader::readAllMessages(
        const std::set<std::string>& topics,
        uint64_t start_time_ns,
        uint64_t end_time_ns) {
    std::vector<BagMessage> messages;
    readMessages(
            [&messages](const BagMessage& msg) {
                messages.push_back(msg);
                return true;
            },
            topics, start_time_ns, end_time_ns);

    std::sort(messages.begin(), messages.end(),
              [](const BagMessage& a, const BagMessage& b) {
                  return a.timestamp_ns < b.timestamp_ns;
              });
    return messages;
}

BagMessage RosBagReader::readFirstMessage(const std::string& topic) {
    BagMessage result;
    std::set<std::string> topics = {topic};
    readMessages(
            [&result](const BagMessage& msg) {
                result = msg;
                return false;
            },
            topics);
    return result;
}

uint64_t RosBagReader::percentToTimestamp(double percent) const {
    percent = std::max(0.0, std::min(1.0, percent));
    if (bag_end_time_ <= bag_begin_time_) return bag_begin_time_;
    return bag_begin_time_ +
           static_cast<uint64_t>((bag_end_time_ - bag_begin_time_) * percent);
}

void RosBagReader::touchChunkCache(size_t chunk_idx) const {
    chunk_cache_lru_.remove(chunk_idx);
    chunk_cache_lru_.push_front(chunk_idx);
}

void RosBagReader::evictChunkCacheIfNeeded() const {
    while (chunk_cache_.size() > chunk_cache_capacity_ &&
           !chunk_cache_lru_.empty()) {
        size_t victim = chunk_cache_lru_.back();
        chunk_cache_lru_.pop_back();
        chunk_cache_.erase(victim);
    }
}

bool RosBagReader::getDecompressedChunk(size_t chunk_idx,
                                        std::vector<char>& out) const {
    if (chunk_idx >= chunk_records_.size()) return false;

    {
        std::lock_guard<std::mutex> lock(chunk_cache_mutex_);
        auto it = chunk_cache_.find(chunk_idx);
        if (it != chunk_cache_.end()) {
            out = it->second;
            touchChunkCache(chunk_idx);
            return true;
        }
    }

    std::vector<char> raw_data;
    {
        std::lock_guard<std::mutex> lock(io_mutex_);
        if (!readChunkRaw(chunk_records_[chunk_idx], raw_data)) {
            return false;
        }
    }

    std::vector<char> decompressed;
    if (!decompressChunk(chunk_records_[chunk_idx], raw_data, decompressed)) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(chunk_cache_mutex_);
        chunk_cache_[chunk_idx] = decompressed;
        touchChunkCache(chunk_idx);
        evictChunkCacheIfNeeded();
        out = chunk_cache_[chunk_idx];
    }
    return true;
}

bool RosBagReader::parseMessageRecord(const char* chunk_data,
                                      uint32_t chunk_size,
                                      uint32_t record_offset,
                                      BagMessage& out) const {
    uint32_t pos = record_offset;
    if (pos + 4 > chunk_size) return false;

    uint32_t header_len = readUint32(chunk_data + pos);
    pos += 4;
    if (pos + header_len > chunk_size) return false;

    auto fields = parseHeader(chunk_data + pos, header_len);
    pos += header_len;

    if (pos + 4 > chunk_size) return false;
    uint32_t data_len = readUint32(chunk_data + pos);
    pos += 4;
    if (pos + data_len > chunk_size) return false;

    if (fields.find("op") == fields.end()) return false;
    if (static_cast<uint8_t>(fields.at("op")[0]) != MSGDATA) return false;

    uint32_t conn_id = 0;
    if (fields.count("conn") && fields["conn"].size() >= 4) {
        conn_id = readUint32(fields["conn"].data());
    }
    if (fields.count("time") && fields["time"].size() >= 8) {
        uint32_t secs = readUint32(fields["time"].data());
        uint32_t nsecs = readUint32(fields["time"].data() + 4);
        out.timestamp_ns = static_cast<uint64_t>(secs) * 1000000000ULL + nsecs;
    }

    if (connections_.count(conn_id)) {
        out.topic = connections_.at(conn_id).topic;
    }
    out.data.assign(chunk_data + pos, data_len);
    return true;
}

bool RosBagReader::indexChunkMessages(
        size_t chunk_idx,
        const char* chunk_data,
        uint32_t chunk_size,
        const std::map<uint32_t, std::string>& conn_to_topic,
        std::map<std::string, std::vector<MessageTimeIndexEntry>>& local_index)
        const {
    uint32_t pos = 0;
    while (pos < chunk_size) {
        uint32_t record_offset = pos;
        if (pos + 4 > chunk_size) break;
        uint32_t header_len = readUint32(chunk_data + pos);
        pos += 4;
        if (pos + header_len > chunk_size) break;

        auto fields = parseHeader(chunk_data + pos, header_len);
        pos += header_len;

        if (pos + 4 > chunk_size) break;
        uint32_t data_len = readUint32(chunk_data + pos);
        pos += 4;
        if (pos + data_len > chunk_size) break;

        if (fields.find("op") != fields.end() &&
            static_cast<uint8_t>(fields["op"][0]) == MSGDATA) {
            uint32_t conn_id = 0;
            uint64_t time_ns = 0;
            if (fields.count("conn") && fields["conn"].size() >= 4) {
                conn_id = readUint32(fields["conn"].data());
            }
            if (fields.count("time") && fields["time"].size() >= 8) {
                uint32_t secs = readUint32(fields["time"].data());
                uint32_t nsecs = readUint32(fields["time"].data() + 4);
                time_ns = static_cast<uint64_t>(secs) * 1000000000ULL + nsecs;
            }

            auto it = conn_to_topic.find(conn_id);
            if (it != conn_to_topic.end()) {
                local_index[it->second].push_back(
                        {time_ns, chunk_idx, record_offset});
            }
        }
        pos += data_len;
    }
    return true;
}

bool RosBagReader::buildTopicTimeIndex(const std::set<std::string>& topics,
                                       std::function<void(int)> progress_cb,
                                       int num_threads) {
    if (!is_open_) return false;

    clearTopicTimeIndex();

    if (multi_mode_) {
        // Group topics by their owning sub-reader so each sub-reader builds
        // its index ONCE with ALL its topics. Calling buildTopicTimeIndex
        // per-topic would clear the sub-reader's index each time, leaving only
        // the last topic indexed — a subtle bug that made
        // readMessageNearestTime return empty data for all-but-one topic.
        std::map<RosBagReader*, std::set<std::string>> topics_by_owner;
        for (const auto& topic : topics) {
            RosBagReader* owner = readerForTopic(topic);
            if (!owner) continue;
            topics_by_owner[owner].insert(topic);
        }
        bool any = false;
        for (auto& [owner, owner_topics] : topics_by_owner) {
            any = owner->buildTopicTimeIndex(owner_topics, progress_cb,
                                             num_threads) ||
                  any;
        }
        index_built_ = any;
        index_ready_.store(any, std::memory_order_release);
        return any;
    }

    std::map<uint32_t, std::string> conn_to_topic;
    for (const auto& topic : topics) {
        auto it = topic_to_conn_.find(topic);
        if (it != topic_to_conn_.end()) {
            conn_to_topic[it->second] = topic;
        }
    }
    if (conn_to_topic.empty()) return false;

    if (num_threads <= 0) {
        num_threads = cloudViewer::utility::CPUInfo::GetInstance().NumThreads();
        if (num_threads <= 0) num_threads = 4;
    }

    struct ChunkWork {
        size_t chunk_idx = 0;
        ChunkRecord record;
        std::vector<char> raw_data;
    };

    const size_t total_chunks = chunk_records_.size();
    const size_t batch_size =
            std::max<size_t>(1, static_cast<size_t>(num_threads) * 4);
    std::atomic<size_t> done_count{0};

    auto process_batch = [&](size_t batch_start, size_t batch_end) {
        std::vector<ChunkWork> work_items;
        work_items.reserve(batch_end - batch_start);
        for (size_t i = batch_start; i < batch_end; ++i) {
            ChunkWork cw;
            cw.chunk_idx = i;
            cw.record = chunk_records_[i];
            {
                std::lock_guard<std::mutex> lock(io_mutex_);
                if (!readChunkRaw(cw.record, cw.raw_data)) continue;
            }
            work_items.push_back(std::move(cw));
        }
        if (work_items.empty()) return;

        std::vector<std::map<std::string, std::vector<MessageTimeIndexEntry>>>
                per_work_index(work_items.size());

        const size_t batch_total = work_items.size();
        auto worker = [&](size_t start, size_t end) {
            for (size_t wi = start; wi < end; ++wi) {
                auto& cw = work_items[wi];
                std::vector<char> decompressed;
                if (!decompressChunk(cw.record, cw.raw_data, decompressed)) {
                    ++done_count;
                    continue;
                }
                cw.raw_data.clear();
                cw.raw_data.shrink_to_fit();

                indexChunkMessages(cw.chunk_idx, decompressed.data(),
                                   static_cast<uint32_t>(decompressed.size()),
                                   conn_to_topic, per_work_index[wi]);

                size_t d = ++done_count;
                if (progress_cb && total_chunks > 0) {
                    progress_cb(static_cast<int>(d * 100 / total_chunks));
                }
            }
        };

        if (num_threads <= 1 || batch_total <= 1) {
            worker(0, batch_total);
        } else {
            size_t n = std::min(static_cast<size_t>(num_threads), batch_total);
            std::vector<std::thread> threads;
            size_t per_thread = (batch_total + n - 1) / n;
            for (size_t t = 0; t < n; ++t) {
                size_t s = t * per_thread;
                size_t e = std::min(s + per_thread, batch_total);
                if (s >= batch_total) break;
                threads.emplace_back(worker, s, e);
            }
            for (auto& th : threads) th.join();
        }

        for (auto& local : per_work_index) {
            for (auto& [topic, entries] : local) {
                auto& dst = topic_time_index_[topic];
                dst.insert(dst.end(), std::make_move_iterator(entries.begin()),
                           std::make_move_iterator(entries.end()));
            }
        }
    };

    for (size_t batch_start = 0; batch_start < total_chunks;
         batch_start += batch_size) {
        const size_t batch_end =
                std::min(batch_start + batch_size, total_chunks);
        process_batch(batch_start, batch_end);
    }

    for (auto& [topic, entries] : topic_time_index_) {
        std::sort(entries.begin(), entries.end(),
                  [](const MessageTimeIndexEntry& a,
                     const MessageTimeIndexEntry& b) {
                      return a.timestamp_ns < b.timestamp_ns;
                  });
        CVLog::Print("[RosBagReader] indexed topic '%s': %zu entries",
                     topic.c_str(), entries.size());
    }

    index_built_ = !topic_time_index_.empty();
    index_ready_.store(index_built_, std::memory_order_release);
    CVLog::Print("[RosBagReader] time index built: %zu topics",
                 topic_time_index_.size());
    return index_built_;
}

BagMessage RosBagReader::readMessageAtPercentIndexed(const std::string& topic,
                                                     double percent) const {
    BagMessage result;
    if (multi_mode_) {
        const uint64_t target = percentToTimestamp(percent);
        return readMessageNearestTime(topic, target);
    }
    if (!index_ready_.load(std::memory_order_acquire)) return result;

    std::lock_guard<std::mutex> lock(index_mutex_);
    if (!index_built_) return result;

    auto it = topic_time_index_.find(topic);
    if (it == topic_time_index_.end() || it->second.empty()) return result;

    const auto& entries = it->second;
    const uint64_t target = percentToTimestamp(percent);

    auto cmp = [](const MessageTimeIndexEntry& e, uint64_t t) {
        return e.timestamp_ns < t;
    };
    auto pos = std::lower_bound(entries.begin(), entries.end(), target, cmp);

    std::vector<const MessageTimeIndexEntry*> candidates;
    if (pos != entries.end()) candidates.push_back(&(*pos));
    if (pos != entries.begin()) candidates.push_back(&(*(pos - 1)));
    if (candidates.empty()) return result;

    const MessageTimeIndexEntry* best_entry = candidates.front();
    uint64_t best_delta = UINT64_MAX;
    for (const auto* entry : candidates) {
        uint64_t d = entry->timestamp_ns > target
                             ? entry->timestamp_ns - target
                             : target - entry->timestamp_ns;
        if (d < best_delta) {
            best_delta = d;
            best_entry = entry;
        }
    }

    std::vector<char> chunk_data;
    if (!getDecompressedChunk(best_entry->chunk_idx, chunk_data)) {
        return result;
    }

    if (!parseMessageRecord(chunk_data.data(),
                            static_cast<uint32_t>(chunk_data.size()),
                            best_entry->record_offset, result)) {
        return {};
    }
    return result;
}

BagMessage RosBagReader::readMessageAtIndexEntry(
        const std::string& topic, const MessageTimeIndexEntry& entry) const {
    BagMessage result;
    if (!is_open_) return result;

    if (multi_mode_) {
        const RosBagReader* owner = readerForTopic(topic);
        if (!owner) return result;
        return owner->readMessageAtIndexEntry(topic, entry);
    }

    if (!index_ready_.load(std::memory_order_acquire)) return result;

    std::vector<char> chunk_data;
    if (!getDecompressedChunk(entry.chunk_idx, chunk_data)) {
        return result;
    }
    if (!parseMessageRecord(chunk_data.data(),
                            static_cast<uint32_t>(chunk_data.size()),
                            entry.record_offset, result)) {
        return {};
    }
    if (result.topic.empty() && topic_to_conn_.count(topic)) {
        result.topic = topic;
    }
    return result;
}

BagMessage RosBagReader::readMessageNearestTime(const std::string& topic,
                                                uint64_t target_ns,
                                                uint64_t max_delta_ns) const {
    BagMessage result;
    if (!is_open_) return result;

    if (multi_mode_) {
        const RosBagReader* owner = readerForTopic(topic);
        if (!owner) return result;
        if (owner->hasTopicTimeIndex()) {
            return owner->readMessageNearestTime(topic, target_ns,
                                                 max_delta_ns);
        }
        BagMessage nearest;
        uint64_t best_delta = UINT64_MAX;
        const_cast<RosBagReader*>(owner)->readMessages(
                [&](const BagMessage& msg) {
                    if (msg.topic != topic) return true;
                    const uint64_t delta =
                            msg.timestamp_ns > target_ns
                                    ? msg.timestamp_ns - target_ns
                                    : target_ns - msg.timestamp_ns;
                    if (delta < best_delta) {
                        best_delta = delta;
                        nearest = msg;
                    }
                    return true;
                },
                {topic},
                target_ns > max_delta_ns ? target_ns - max_delta_ns : 0,
                target_ns + max_delta_ns);
        if (best_delta <= max_delta_ns) return nearest;
        return {};
    }

    if (!index_ready_.load(std::memory_order_acquire)) return result;

    MessageTimeIndexEntry entry{};
    {
        std::lock_guard<std::mutex> lock(index_mutex_);
        if (!index_built_) return result;

        auto it = topic_time_index_.find(topic);
        if (it == topic_time_index_.end() || it->second.empty()) return result;

        const auto& entries = it->second;
        auto cmp = [](const MessageTimeIndexEntry& e, uint64_t t) {
            return e.timestamp_ns < t;
        };
        auto pos = std::lower_bound(entries.begin(), entries.end(), target_ns,
                                    cmp);

        std::vector<const MessageTimeIndexEntry*> candidates;
        if (pos != entries.end()) candidates.push_back(&(*pos));
        if (pos != entries.begin()) candidates.push_back(&(*(pos - 1)));
        if (candidates.empty()) return result;

        const MessageTimeIndexEntry* best_entry = candidates.front();
        uint64_t best_delta = UINT64_MAX;
        for (const auto* candidate : candidates) {
            uint64_t d = candidate->timestamp_ns > target_ns
                                 ? candidate->timestamp_ns - target_ns
                                 : target_ns - candidate->timestamp_ns;
            if (d < best_delta) {
                best_delta = d;
                best_entry = candidate;
            }
        }
        if (best_delta > max_delta_ns) return result;
        entry = *best_entry;
    }
    return readMessageAtIndexEntry(topic, entry);
}

std::vector<MessageTimeIndexEntry> RosBagReader::getMessageTimeline(
        const std::string& topic) const {
    std::vector<MessageTimeIndexEntry> out;
    if (!is_open_) return out;

    if (multi_mode_) {
        // Each topic is owned by exactly one sub-bag; aggregate from there.
        const RosBagReader* owner = readerForTopic(topic);
        if (!owner) return out;
        return owner->getMessageTimeline(topic);
    }

    if (!index_ready_.load(std::memory_order_acquire)) return out;
    std::lock_guard<std::mutex> lock(index_mutex_);
    if (!index_built_) return out;

    auto it = topic_time_index_.find(topic);
    if (it == topic_time_index_.end()) return out;
    out = it->second;  // copy (sorted by buildTopicTimeIndex)
    return out;
}

bool RosBagReader::findNearestIndexEntry(const std::string& topic,
                                         uint64_t target_ns,
                                         MessageTimeIndexEntry& out_entry,
                                         uint64_t max_delta_ns) const {
    out_entry = MessageTimeIndexEntry{};
    if (!is_open_) return false;

    if (multi_mode_) {
        const RosBagReader* owner = readerForTopic(topic);
        if (!owner) return false;
        return owner->findNearestIndexEntry(topic, target_ns, out_entry,
                                            max_delta_ns);
    }

    if (!index_ready_.load(std::memory_order_acquire)) return false;

    std::lock_guard<std::mutex> lock(index_mutex_);
    if (!index_built_) return false;

    auto it = topic_time_index_.find(topic);
    if (it == topic_time_index_.end() || it->second.empty()) return false;

    const auto& entries = it->second;
    auto cmp = [](const MessageTimeIndexEntry& e, uint64_t t) {
        return e.timestamp_ns < t;
    };
    auto pos = std::lower_bound(entries.begin(), entries.end(), target_ns, cmp);

    std::vector<const MessageTimeIndexEntry*> candidates;
    if (pos != entries.end()) candidates.push_back(&(*pos));
    if (pos != entries.begin()) candidates.push_back(&(*(pos - 1)));
    if (candidates.empty()) return false;

    const MessageTimeIndexEntry* best_entry = candidates.front();
    uint64_t best_delta = UINT64_MAX;
    for (const auto* candidate : candidates) {
        uint64_t d = candidate->timestamp_ns > target_ns
                             ? candidate->timestamp_ns - target_ns
                             : target_ns - candidate->timestamp_ns;
        if (d < best_delta) {
            best_delta = d;
            best_entry = candidate;
        }
    }
    if (best_delta > max_delta_ns) return false;
    out_entry = *best_entry;
    return true;
}

std::vector<BagMessage> RosBagReader::readMessagesAtPercentParallel(
        const std::vector<std::string>& topics,
        double percent,
        int num_threads) {
    std::vector<BagMessage> results(topics.size());
    if (topics.empty() || !is_open_) return results;

    if (multi_mode_) {
        if (num_threads <= 0) {
            num_threads =
                    cloudViewer::utility::CPUInfo::GetInstance().NumThreads();
            if (num_threads <= 0) num_threads = 4;
        }

        // Critical: compute target_ns from the WRAPPER's (refined) playback
        // range, not each sub-bag's raw range. Sub-bag percentToTimestamp
        // would map percent=0.5 to different absolute times across bags,
        // breaking 25ms multi-camera sync in tryGetAlignedImagesIndexed.
        const uint64_t target_ns = percentToTimestamp(percent);

        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                const RosBagReader* owner = readerForTopic(topics[i]);
                if (owner && owner->hasTopicTimeIndex()) {
                    // Use readMessageNearestTime (target_ns based) so all
                    // sub-bags resolve the same absolute playback time.
                    results[i] =
                            owner->readMessageNearestTime(topics[i], target_ns);
                } else {
                    results[i] = readMessageAtPercent(topics[i], percent);
                }
                if (results[i].topic.empty()) {
                    results[i].topic = topics[i];
                }
            }
        };

        if (num_threads <= 1 || topics.size() <= 1) {
            worker(0, topics.size());
        } else {
            const size_t n =
                    std::min(static_cast<size_t>(num_threads), topics.size());
            std::vector<std::thread> threads;
            const size_t per_thread = (topics.size() + n - 1) / n;
            for (size_t t = 0; t < n; ++t) {
                const size_t s = t * per_thread;
                const size_t e = std::min(s + per_thread, topics.size());
                if (s >= topics.size()) break;
                threads.emplace_back(worker, s, e);
            }
            for (auto& th : threads) th.join();
        }
        return results;
    }

    std::map<std::string, uint32_t> topic_conn;
    std::set<uint32_t> conn_ids;
    for (size_t i = 0; i < topics.size(); ++i) {
        results[i].topic = topics[i];
        auto it = topic_to_conn_.find(topics[i]);
        if (it != topic_to_conn_.end()) {
            topic_conn[topics[i]] = it->second;
            conn_ids.insert(it->second);
        }
    }
    if (conn_ids.empty()) return results;

    if (index_ready_.load(std::memory_order_acquire)) {
        if (num_threads <= 0) {
            num_threads =
                    cloudViewer::utility::CPUInfo::GetInstance().NumThreads();
            if (num_threads <= 0) num_threads = 4;
        }

        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                results[i] = readMessageAtPercentIndexed(topics[i], percent);
            }
        };

        if (num_threads <= 1 || topics.size() <= 1) {
            worker(0, topics.size());
        } else {
            size_t n =
                    std::min(static_cast<size_t>(num_threads), topics.size());
            std::vector<std::thread> threads;
            size_t per_thread = (topics.size() + n - 1) / n;
            for (size_t t = 0; t < n; ++t) {
                size_t s = t * per_thread;
                size_t e = std::min(s + per_thread, topics.size());
                if (s >= topics.size()) break;
                threads.emplace_back(worker, s, e);
            }
            for (auto& th : threads) th.join();
        }
        return results;
    }

    const uint64_t target_time = percentToTimestamp(percent);
    const auto chunk_candidates = findCandidateChunks(target_time);

    std::map<uint32_t, BagMessage> best_by_conn;
    std::map<uint32_t, uint64_t> best_delta_by_conn;

    for (size_t chunk_idx : chunk_candidates) {
        std::vector<char> chunk_data;
        if (!getDecompressedChunk(chunk_idx, chunk_data)) continue;

        scanChunkForTopics(
                chunk_data.data(), static_cast<uint32_t>(chunk_data.size()),
                target_time, conn_ids, best_by_conn, best_delta_by_conn);
    }

    if (best_by_conn.size() < conn_ids.size()) {
        for (size_t chunk_idx = 0; chunk_idx < chunk_records_.size();
             ++chunk_idx) {
            bool already = false;
            for (size_t c : chunk_candidates) {
                if (c == chunk_idx) {
                    already = true;
                    break;
                }
            }
            if (already) continue;

            std::vector<char> chunk_data;
            if (!getDecompressedChunk(chunk_idx, chunk_data)) continue;
            scanChunkForTopics(
                    chunk_data.data(), static_cast<uint32_t>(chunk_data.size()),
                    target_time, conn_ids, best_by_conn, best_delta_by_conn);
            if (best_by_conn.size() == conn_ids.size()) break;
        }
    }

    for (size_t i = 0; i < topics.size(); ++i) {
        auto conn_it = topic_conn.find(topics[i]);
        if (conn_it == topic_conn.end()) continue;
        auto msg_it = best_by_conn.find(conn_it->second);
        if (msg_it != best_by_conn.end()) {
            results[i] = msg_it->second;
        }
    }
    return results;
}

std::vector<size_t> RosBagReader::findCandidateChunks(
        uint64_t target_time_ns) const {
    std::vector<size_t> out;
    if (chunk_infos_.empty()) {
        if (!chunk_records_.empty()) out.push_back(0);
        return out;
    }

    auto cmp = [](const ChunkInfo& ci, uint64_t t) { return ci.end_time < t; };
    auto pos = std::lower_bound(chunk_infos_.begin(), chunk_infos_.end(),
                                target_time_ns, cmp);

    if (pos != chunk_infos_.end()) {
        out.push_back(static_cast<size_t>(pos - chunk_infos_.begin()));
    }
    if (pos != chunk_infos_.begin()) {
        out.push_back(static_cast<size_t>(pos - chunk_infos_.begin() - 1));
    }
    if (out.empty()) out.push_back(0);
    return out;
}

void RosBagReader::scanChunkForTopics(
        const char* chunk_data,
        uint32_t chunk_size,
        uint64_t target_time_ns,
        const std::set<uint32_t>& conn_ids,
        std::map<uint32_t, BagMessage>& best_by_conn,
        std::map<uint32_t, uint64_t>& best_delta_by_conn) const {
    uint32_t pos = 0;
    while (pos < chunk_size) {
        if (pos + 4 > chunk_size) break;
        uint32_t header_len = readUint32(chunk_data + pos);
        pos += 4;
        if (pos + header_len > chunk_size) break;

        auto fields = parseHeader(chunk_data + pos, header_len);
        pos += header_len;

        if (pos + 4 > chunk_size) break;
        uint32_t data_len = readUint32(chunk_data + pos);
        pos += 4;
        if (pos + data_len > chunk_size) break;

        if (fields.find("op") != fields.end() &&
            static_cast<uint8_t>(fields["op"][0]) == MSGDATA) {
            uint32_t conn_id = 0;
            uint64_t time_ns = 0;
            if (fields.count("conn") && fields["conn"].size() >= 4) {
                conn_id = readUint32(fields["conn"].data());
            }
            if (fields.count("time") && fields["time"].size() >= 8) {
                uint32_t secs = readUint32(fields["time"].data());
                uint32_t nsecs = readUint32(fields["time"].data() + 4);
                time_ns = static_cast<uint64_t>(secs) * 1000000000ULL + nsecs;
            }

            if (conn_ids.count(conn_id)) {
                uint64_t delta = time_ns > target_time_ns
                                         ? time_ns - target_time_ns
                                         : target_time_ns - time_ns;
                auto it = best_delta_by_conn.find(conn_id);
                if (it == best_delta_by_conn.end() || delta < it->second) {
                    BagMessage msg;
                    if (connections_.count(conn_id)) {
                        msg.topic = connections_.at(conn_id).topic;
                    }
                    msg.timestamp_ns = time_ns;
                    msg.data.assign(chunk_data + pos, data_len);
                    best_by_conn[conn_id] = std::move(msg);
                    best_delta_by_conn[conn_id] = delta;
                }
            }
        }
        pos += data_len;
    }
}

BagMessage RosBagReader::readMessageAtPercent(const std::string& topic,
                                              double percent) {
    if (multi_mode_) {
        if (index_ready_.load(std::memory_order_acquire)) {
            return readMessageAtPercentIndexed(topic, percent);
        }
        const RosBagReader* owner = readerForTopic(topic);
        if (!owner) return {};
        const uint64_t target = percentToTimestamp(percent);
        BagMessage nearest = readMessageNearestTime(topic, target, UINT64_MAX);
        if (!nearest.data.empty()) return nearest;
        return const_cast<RosBagReader*>(owner)->readFirstMessage(topic);
    }

    if (index_ready_.load(std::memory_order_acquire)) {
        return readMessageAtPercentIndexed(topic, percent);
    }

    auto results = readMessagesAtPercentParallel({topic}, percent);
    if (!results.empty()) return results.front();

    percent = std::max(0.0, std::min(1.0, percent));

    if (bag_end_time_ <= bag_begin_time_) {
        CVLog::Warning(
                "[RosBagReader] readAtPercent: invalid time range "
                "(begin=%lu, end=%lu), using readFirstMessage",
                static_cast<unsigned long>(bag_begin_time_),
                static_cast<unsigned long>(bag_end_time_));
        return readFirstMessage(topic);
    }

    uint64_t target_time =
            bag_begin_time_ +
            static_cast<uint64_t>((bag_end_time_ - bag_begin_time_) * percent);

    uint64_t search_window = 1000000000ULL;  // 1 second
    uint64_t start =
            target_time > search_window ? target_time - search_window : 0;
    uint64_t end = target_time + search_window;

    BagMessage best;
    uint64_t best_delta = UINT64_MAX;
    std::set<std::string> topics_set = {topic};

    auto it = topic_to_conn_.find(topic);
    if (it == topic_to_conn_.end()) {
        CVLog::Warning(
                "[RosBagReader] readAtPercent: topic '%s' NOT in bag "
                "connections!",
                topic.c_str());
        return best;
    }

    readMessages(
            [&](const BagMessage& msg) {
                uint64_t d = msg.timestamp_ns > target_time
                                     ? msg.timestamp_ns - target_time
                                     : target_time - msg.timestamp_ns;
                if (d < best_delta) {
                    best_delta = d;
                    best = msg;
                }
                return true;
            },
            topics_set, start, end);

    if (best.data.empty()) {
        search_window = 5000000000ULL;  // 5 seconds fallback
        start = target_time > search_window ? target_time - search_window : 0;
        end = target_time + search_window;
        readMessages(
                [&](const BagMessage& msg) {
                    uint64_t d = msg.timestamp_ns > target_time
                                         ? msg.timestamp_ns - target_time
                                         : target_time - msg.timestamp_ns;
                    if (d < best_delta) {
                        best_delta = d;
                        best = msg;
                    }
                    return true;
                },
                topics_set, start, end);
    }

    if (best.data.empty()) {
        CVLog::Warning(
                "[RosBagReader] readAtPercent: no message found for '%s' "
                "at %.1f%% (target_ns=%lu)",
                topic.c_str(), percent * 100.0,
                static_cast<unsigned long>(target_time));
    }

    return best;
}

bool RosBagReader::decompressBZ2(const char* input,
                                 uint32_t input_len,
                                 uint32_t uncompressed_size,
                                 std::vector<char>& output) {
#ifdef MCALIB_HAS_BZ2
    unsigned int out_size =
            uncompressed_size > 0 ? uncompressed_size : input_len * 10;
    output.resize(out_size);

    int result = BZ2_bzBuffToBuffDecompress(output.data(), &out_size,
                                            const_cast<char*>(input), input_len,
                                            0, 0);

    if (result == BZ_OUTBUFF_FULL && uncompressed_size == 0) {
        out_size = static_cast<unsigned int>(output.size() * 4);
        output.resize(out_size);
        result = BZ2_bzBuffToBuffDecompress(output.data(), &out_size,
                                            const_cast<char*>(input), input_len,
                                            0, 0);
    }

    if (result != BZ_OK) {
        CVLog::Warning("[RosBagReader] BZ2 decompression error: %d", result);
        return false;
    }

    output.resize(out_size);
    return true;
#else
    CVLog::Warning(
            "[RosBagReader] BZ2 decompression not available. "
            "Rebuild with libbz2-dev.");
    return false;
#endif
}

bool RosBagReader::decompressLZ4(const char* input,
                                 uint32_t input_len,
                                 uint32_t uncompressed_size,
                                 std::vector<char>& output) {
#ifdef MCALIB_HAS_LZ4
    if (input_len < 8) {
        CVLog::Warning("[RosBagReader] LZ4 data too short (%u bytes)",
                       input_len);
        return false;
    }

    // roslz4 stream format:
    //   Header: block_size(u32 LE) + reserved(u32 LE)
    //   Blocks: compressed_block_size(u32 LE) + data
    uint32_t block_size;
    std::memcpy(&block_size, input, 4);

    if (block_size == 0 || block_size > 64 * 1024 * 1024) {
        CVLog::Warning("[RosBagReader] Invalid LZ4 block size: %u", block_size);
        return false;
    }

    uint32_t out_cap =
            uncompressed_size > 0 ? uncompressed_size : block_size * 16;
    output.resize(out_cap);

    uint32_t in_pos = 8;
    uint32_t out_pos = 0;

    while (in_pos < input_len) {
        if (in_pos + 4 > input_len) break;

        uint32_t comp_block_size;
        std::memcpy(&comp_block_size, input + in_pos, 4);
        in_pos += 4;

        if (comp_block_size == 0) break;
        if (in_pos + comp_block_size > input_len) {
            CVLog::Warning(
                    "[RosBagReader] LZ4 block overruns input "
                    "(need %u, have %u)",
                    comp_block_size, input_len - in_pos);
            break;
        }

        if (out_pos + block_size > output.size()) {
            output.resize(out_pos + block_size * 2);
        }

        int decompressed =
                LZ4_decompress_safe(input + in_pos, output.data() + out_pos,
                                    static_cast<int>(comp_block_size),
                                    static_cast<int>(output.size() - out_pos));

        if (decompressed < 0) {
            CVLog::Warning(
                    "[RosBagReader] LZ4_decompress_safe failed at "
                    "block offset %u (error=%d)",
                    in_pos, decompressed);
            return false;
        }

        out_pos += static_cast<uint32_t>(decompressed);
        in_pos += comp_block_size;
    }

    output.resize(out_pos);

    if (uncompressed_size > 0 && out_pos != uncompressed_size) {
        CVLog::Warning(
                "[RosBagReader] LZ4 output size mismatch: expected %u, "
                "got %u",
                uncompressed_size, out_pos);
    }

    return out_pos > 0;
#else
    CVLog::Warning(
            "[RosBagReader] LZ4 decompression not available. "
            "Rebuild with liblz4-dev.");
    return false;
#endif
}

}  // namespace mcalib
