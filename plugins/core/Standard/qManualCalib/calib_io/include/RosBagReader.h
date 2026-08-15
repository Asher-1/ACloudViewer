// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace mcalib {

struct BagMessage {
    std::string topic;
    uint64_t timestamp_ns = 0;
    std::string data;

    std::string getStringPayload() const;
};

struct BagConnection {
    uint32_t id = 0;
    std::string topic;
    std::string datatype;
    std::string md5sum;
    std::string message_definition;
};

struct MessageTimeIndexEntry {
    uint64_t timestamp_ns = 0;
    size_t chunk_idx = 0;
    uint32_t record_offset = 0;
};

class VideoDecodeCache;

class RosBagReader {
public:
    RosBagReader();
    ~RosBagReader();

    bool open(const std::string& path);

    /// Open multiple bags as one logical timeline (Heavy/Light/Medium groups).
    /// Topics are routed to their source bag; playback uses unified timestamps.
    bool openMulti(const std::vector<std::string>& paths);

    void close();
    bool isOpen() const;

    std::string getPath() const { return path_; }
    const std::vector<std::string>& getSourceBagPaths() const {
        return source_bag_paths_;
    }
    bool isMultiBagMode() const { return multi_mode_; }

    uint64_t getBeginTime() const;
    uint64_t getEndTime() const;
    uint64_t getDuration() const;

    std::vector<std::string> getTopics() const;
    std::map<std::string, std::string> getTopicTypes() const;
    const std::map<uint32_t, BagConnection>& getConnections() const {
        return connections_;
    }
    uint32_t getMessageCount() const;

    using MessageCallback = std::function<bool(const BagMessage&)>;

    bool readMessages(const MessageCallback& callback,
                      const std::set<std::string>& topics = {},
                      uint64_t start_time_ns = 0,
                      uint64_t end_time_ns = UINT64_MAX);

    std::vector<BagMessage> readAllMessages(
            const std::set<std::string>& topics = {},
            uint64_t start_time_ns = 0,
            uint64_t end_time_ns = UINT64_MAX);

    std::vector<BagMessage> readAllMessagesParallel(
            const std::set<std::string>& topics = {},
            std::function<void(int)> progress_cb = nullptr,
            uint64_t start_time_ns = 0,
            uint64_t end_time_ns = UINT64_MAX,
            int num_threads = 4);

    BagMessage readFirstMessage(const std::string& topic);
    BagMessage readMessageAtPercent(const std::string& topic, double percent);

    /// Nearest indexed message to target time (requires topic time index).
    BagMessage readMessageNearestTime(const std::string& topic,
                                      uint64_t target_ns,
                                      uint64_t max_delta_ns = UINT64_MAX) const;

    /// Load one indexed message by chunk/offset (requires topic time index).
    BagMessage readMessageAtIndexEntry(
            const std::string& topic, const MessageTimeIndexEntry& entry) const;

    /// Narrow playback range using first/last timestamp of a reference camera
    /// topic.
    bool refinePlaybackTimeRange(const std::string& topic);

    /// Persistent H.264/HEVC decoder state (cleared on close).
    VideoDecodeCache& videoDecodeCache();
    void clearVideoDecodeCache();

    /// Build lightweight per-topic timestamp index (no message payloads).
    bool buildTopicTimeIndex(const std::set<std::string>& topics,
                             std::function<void(int)> progress_cb = nullptr,
                             int num_threads = 0);
    bool hasTopicTimeIndex() const {
        return index_ready_.load(std::memory_order_acquire);
    }
    void clearTopicTimeIndex();

    /// Parallel seek for multiple topics at the same timeline percent.
    std::vector<BagMessage> readMessagesAtPercentParallel(
            const std::vector<std::string>& topics,
            double percent,
            int num_threads = 0);

    /// Read-only view of a topic's timestamp index (empty if not built).
    /// Returned by value so callers can binary-search without holding the
    /// index mutex. Multi-bag mode aggregates entries from every sub-bag that
    /// owns the topic.
    std::vector<MessageTimeIndexEntry> getMessageTimeline(
            const std::string& topic) const;

    /// Indexed seek for a single message nearest to target_ns. Returns true
    /// and fills out_entry on success. Requires topic time index.
    bool findNearestIndexEntry(const std::string& topic,
                               uint64_t target_ns,
                               MessageTimeIndexEntry& out_entry,
                               uint64_t max_delta_ns = UINT64_MAX) const;

    /// Convert playback percent [0,1] to absolute bag timestamp using the
    /// (possibly refined) playback range. Public so callers (e.g. BagAlignment)
    /// can binary-search the topic timeline consistently with the slider.
    uint64_t percentToTimestampPublic(double percent) const {
        return percentToTimestamp(percent);
    }

    static uint32_t readUint32(const char* data);
    static uint64_t readUint64(const char* data);

private:
    struct ChunkInfo {
        uint64_t chunk_pos = 0;
        uint64_t start_time = 0;
        uint64_t end_time = 0;
        uint32_t count = 0;
    };

    struct ChunkRecord {
        uint64_t data_pos = 0;
        std::string compression;
        uint32_t uncompressed_size = 0;
        uint32_t compressed_size = 0;
    };

    struct IndexEntry {
        uint64_t time_ns = 0;
        uint32_t offset = 0;
    };

    bool readVersion();
    bool readBagHeader();
    bool readRecords();
    bool readChunkData(const ChunkRecord& cr,
                       const std::set<uint32_t>& conn_ids,
                       uint64_t start_time_ns,
                       uint64_t end_time_ns,
                       const MessageCallback& callback);

    bool readChunkRaw(const ChunkRecord& cr, std::vector<char>& raw_data) const;
    bool decompressChunk(const ChunkRecord& cr,
                         const std::vector<char>& raw_data,
                         std::vector<char>& decompressed) const;

    bool parseChunkMessages(const char* chunk_data,
                            uint32_t chunk_size,
                            const std::set<uint32_t>& conn_ids,
                            uint64_t start_time_ns,
                            uint64_t end_time_ns,
                            const MessageCallback& callback);

    std::vector<BagMessage> parseChunkMessagesToVec(
            const char* chunk_data,
            uint32_t chunk_size,
            const std::set<uint32_t>& conn_ids,
            uint64_t start_time_ns,
            uint64_t end_time_ns);

    static std::map<std::string, std::string> parseHeader(const char* data,
                                                          uint32_t len);

    static bool decompressBZ2(const char* input,
                              uint32_t input_len,
                              uint32_t uncompressed_size,
                              std::vector<char>& output);
    static bool decompressLZ4(const char* input,
                              uint32_t input_len,
                              uint32_t uncompressed_size,
                              std::vector<char>& output);

    bool indexChunkMessages(
            size_t chunk_idx,
            const char* chunk_data,
            uint32_t chunk_size,
            const std::map<uint32_t, std::string>& conn_to_topic,
            std::map<std::string, std::vector<MessageTimeIndexEntry>>&
                    local_index) const;

    bool parseMessageRecord(const char* chunk_data,
                            uint32_t chunk_size,
                            uint32_t record_offset,
                            BagMessage& out) const;

    bool getDecompressedChunk(size_t chunk_idx, std::vector<char>& out) const;

    void touchChunkCache(size_t chunk_idx) const;
    void evictChunkCacheIfNeeded() const;

    BagMessage readMessageAtPercentIndexed(const std::string& topic,
                                           double percent) const;

    std::vector<size_t> findCandidateChunks(uint64_t target_time_ns) const;

    void scanChunkForTopics(
            const char* chunk_data,
            uint32_t chunk_size,
            uint64_t target_time_ns,
            const std::set<uint32_t>& conn_ids,
            std::map<uint32_t, BagMessage>& best_by_conn,
            std::map<uint32_t, uint64_t>& best_delta_by_conn) const;

    uint64_t percentToTimestamp(double percent) const;

    RosBagReader* readerForTopic(const std::string& topic);
    const RosBagReader* readerForTopic(const std::string& topic) const;
    void resetMultiState();

    struct SubBagReader {
        std::unique_ptr<RosBagReader> reader;
        std::string path;
    };

    mutable std::ifstream file_;
    std::string path_;
    std::vector<std::string> source_bag_paths_;
    std::vector<SubBagReader> sub_readers_;
    std::map<std::string, size_t> topic_owner_idx_;
    bool multi_mode_ = false;
    bool is_open_ = false;

    uint64_t bag_begin_time_ = UINT64_MAX;
    uint64_t bag_end_time_ = 0;
    uint32_t total_message_count_ = 0;

    std::map<uint32_t, BagConnection> connections_;
    std::map<std::string, uint32_t> topic_to_conn_;
    std::vector<ChunkInfo> chunk_infos_;
    std::vector<ChunkRecord> chunk_records_;

    bool index_built_ = false;
    std::atomic<bool> index_ready_{false};
    mutable std::mutex index_mutex_;
    std::map<std::string, std::vector<MessageTimeIndexEntry>> topic_time_index_;

    mutable std::mutex io_mutex_;
    mutable std::mutex chunk_cache_mutex_;
    mutable std::map<size_t, std::vector<char>> chunk_cache_;
    mutable std::list<size_t> chunk_cache_lru_;
    size_t chunk_cache_capacity_ = 32;

    uint64_t index_pos_ = 0;
    uint32_t chunk_count_ = 0;
    uint32_t conn_count_ = 0;

    std::unique_ptr<VideoDecodeCache> video_decode_cache_;
};

}  // namespace mcalib
