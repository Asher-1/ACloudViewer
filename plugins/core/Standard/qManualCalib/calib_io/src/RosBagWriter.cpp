// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RosBagWriter.h"

#include <CVLog.h>

#include <algorithm>
#include <cstring>
#include <fstream>

#include "mcalib_portability.h"

namespace mcalib {

namespace {

constexpr const char* kRosbagMagic = "#ROSBAG V2.0\n";
constexpr int kRosbagMagicLen = 13;

enum RecordOp : uint8_t {
    kMsgData = 0x02,
    kBagHeader = 0x03,
    kIndexData = 0x04,
    kChunk = 0x05,
    kChunkInfo = 0x06,
    kConnection = 0x07
};

void appendUint32(std::vector<char>& buf, uint32_t v) {
    const char* p = reinterpret_cast<const char*>(&v);
    buf.insert(buf.end(), p, p + 4);
}

void appendUint64(std::vector<char>& buf, uint64_t v) {
    const char* p = reinterpret_cast<const char*>(&v);
    buf.insert(buf.end(), p, p + 8);
}

void appendField(std::vector<char>& buf,
                 const std::string& key,
                 const std::string& value) {
    const std::string field = key + "=" + value;
    appendUint32(buf, static_cast<uint32_t>(field.size()));
    buf.insert(buf.end(), field.begin(), field.end());
}

void appendTimeField(std::vector<char>& buf,
                     const std::string& key,
                     uint64_t time_ns) {
    std::string val(8, '\0');
    const uint32_t secs = static_cast<uint32_t>(time_ns / 1000000000ULL);
    const uint32_t nsecs = static_cast<uint32_t>(time_ns % 1000000000ULL);
    std::memcpy(val.data(), &secs, 4);
    std::memcpy(val.data() + 4, &nsecs, 4);
    appendField(buf, key, val);
}

void writeRecord(std::ofstream& out,
                 const std::vector<char>& header,
                 const char* data,
                 uint32_t data_len) {
    const uint32_t header_len = static_cast<uint32_t>(header.size());
    out.write(reinterpret_cast<const char*>(&header_len), 4);
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char*>(&data_len), 4);
    if (data_len > 0 && data) {
        out.write(data, data_len);
    }
}

std::vector<char> buildHeader(
        const std::map<std::string, std::string>& fields) {
    std::vector<char> header;
    for (const auto& [k, v] : fields) {
        appendField(header, k, v);
    }
    return header;
}

}  // namespace

RosBagWriter::RosBagWriter() = default;
RosBagWriter::~RosBagWriter() { close(); }

bool RosBagWriter::open(const std::string& path) {
    close();
    path_ = path;
    is_open_ = true;
    return true;
}

uint32_t RosBagWriter::addConnection(const BagConnection& conn) {
    auto it = topic_to_conn_.find(conn.topic);
    if (it != topic_to_conn_.end()) {
        return it->second;
    }
    const uint32_t id = next_conn_id_++;
    topic_to_conn_[conn.topic] = id;
    connections_[id] = conn;
    connections_[id].id = id;
    return id;
}

bool RosBagWriter::writeMessage(uint32_t conn_id,
                                uint64_t timestamp_ns,
                                const std::string& data) {
    if (!is_open_) return false;
    PendingMessage msg;
    msg.conn_id = conn_id;
    msg.timestamp_ns = timestamp_ns;
    msg.data = data;
    messages_.push_back(std::move(msg));
    return true;
}

bool RosBagWriter::close() {
    if (!is_open_) return true;
    is_open_ = false;

    if (path_.empty()) return false;

    std::ofstream out;
    if (!openOutputFile(out, path_)) {
        CVLog::Error("[RosBagWriter] cannot open %s", path_.c_str());
        return false;
    }

    const size_t msg_count = messages_.size();
    const size_t conn_count = connections_.size();

    out.write(kRosbagMagic, kRosbagMagicLen);

    const uint64_t header_record_pos = static_cast<uint64_t>(out.tellp());
    {
        std::map<std::string, std::string> fields;
        fields["op"] = std::string(1, static_cast<char>(kBagHeader));
        fields["index_pos"] = std::string(8, '\0');
        fields["conn_count"] = std::string(4, '\0');
        fields["chunk_count"] = std::string(4, '\0');
        writeRecord(out, buildHeader(fields), nullptr, 0);
    }

    for (const auto& [id, conn] : connections_) {
        std::map<std::string, std::string> fields;
        fields["op"] = std::string(1, static_cast<char>(kConnection));
        std::string conn_bytes(4, '\0');
        std::memcpy(conn_bytes.data(), &id, 4);
        fields["conn"] = conn_bytes;
        fields["topic"] = conn.topic;

        std::vector<char> body;
        appendField(body, "topic", conn.topic);
        appendField(body, "type", conn.datatype);
        appendField(body, "md5sum", conn.md5sum);
        appendField(body, "message_definition", conn.message_definition);
        writeRecord(out, buildHeader(fields), body.data(),
                    static_cast<uint32_t>(body.size()));
    }

    std::sort(messages_.begin(), messages_.end(),
              [](const PendingMessage& a, const PendingMessage& b) {
                  return a.timestamp_ns < b.timestamp_ns;
              });

    const uint64_t chunk_pos = static_cast<uint64_t>(out.tellp());
    std::vector<char> chunk_data;
    uint64_t begin_time = UINT64_MAX;
    uint64_t end_time = 0;

    for (const auto& msg : messages_) {
        begin_time = std::min(begin_time, msg.timestamp_ns);
        end_time = std::max(end_time, msg.timestamp_ns);

        std::vector<char> msg_header;
        appendField(msg_header, "op",
                    std::string(1, static_cast<char>(kMsgData)));
        std::string conn_bytes(4, '\0');
        std::memcpy(conn_bytes.data(), &msg.conn_id, 4);
        appendField(msg_header, "conn", conn_bytes);
        appendTimeField(msg_header, "time", msg.timestamp_ns);

        appendUint32(chunk_data, static_cast<uint32_t>(msg_header.size()));
        chunk_data.insert(chunk_data.end(), msg_header.begin(),
                          msg_header.end());
        const uint32_t data_len = static_cast<uint32_t>(msg.data.size());
        appendUint32(chunk_data, data_len);
        chunk_data.insert(chunk_data.end(), msg.data.begin(), msg.data.end());
    }

    if (begin_time == UINT64_MAX) begin_time = 0;

    {
        std::map<std::string, std::string> fields;
        fields["op"] = std::string(1, static_cast<char>(kChunk));
        fields["compression"] = "none";
        std::string size_bytes(4, '\0');
        const uint32_t chunk_size = static_cast<uint32_t>(chunk_data.size());
        std::memcpy(size_bytes.data(), &chunk_size, 4);
        fields["size"] = size_bytes;
        writeRecord(out, buildHeader(fields), chunk_data.data(), chunk_size);
    }

    {
        std::vector<char> ci_header;
        appendField(ci_header, "op",
                    std::string(1, static_cast<char>(kChunkInfo)));
        std::string chunk_pos_bytes(8, '\0');
        std::memcpy(chunk_pos_bytes.data(), &chunk_pos, 8);
        appendField(ci_header, "chunk_pos", chunk_pos_bytes);
        appendTimeField(ci_header, "start_time", begin_time);
        appendTimeField(ci_header, "end_time", end_time);
        std::string count_bytes(4, '\0');
        const uint32_t count = static_cast<uint32_t>(messages_.size());
        std::memcpy(count_bytes.data(), &count, 4);
        appendField(ci_header, "count", count_bytes);
        writeRecord(out, ci_header, nullptr, 0);
    }

    const uint64_t index_pos = static_cast<uint64_t>(out.tellp());
    {
        std::vector<char> index_header;
        appendField(index_header, "op",
                    std::string(1, static_cast<char>(kIndexData)));
        writeRecord(out, index_header, nullptr, 0);
    }

    out.seekp(static_cast<std::streamoff>(header_record_pos));
    {
        std::vector<char> bag_header;
        appendField(bag_header, "op",
                    std::string(1, static_cast<char>(kBagHeader)));
        std::string index_bytes(8, '\0');
        std::memcpy(index_bytes.data(), &index_pos, 8);
        appendField(bag_header, "index_pos", index_bytes);
        std::string conn_count_bytes(4, '\0');
        const uint32_t conn_count = static_cast<uint32_t>(connections_.size());
        std::memcpy(conn_count_bytes.data(), &conn_count, 4);
        appendField(bag_header, "conn_count", conn_count_bytes);
        std::string chunk_count_bytes(4, '\0');
        const uint32_t chunk_count = 1;
        std::memcpy(chunk_count_bytes.data(), &chunk_count, 4);
        appendField(bag_header, "chunk_count", chunk_count_bytes);
        const uint32_t header_len = static_cast<uint32_t>(bag_header.size());
        out.write(reinterpret_cast<const char*>(&header_len), 4);
        out.write(bag_header.data(), bag_header.size());
        const uint32_t zero = 0;
        out.write(reinterpret_cast<const char*>(&zero), 4);
    }

    out.close();
    messages_.clear();
    connections_.clear();
    topic_to_conn_.clear();
    next_conn_id_ = 0;

    CVLog::Print("[RosBagWriter] wrote %s (%zu messages, %zu connections)",
                 path_.c_str(), msg_count, conn_count);
    return true;
}

bool mergeRosBags(const std::vector<std::string>& input_bags,
                  const std::string& output_bag) {
    if (input_bags.empty()) return false;

    RosBagWriter writer;
    if (!writer.open(output_bag)) return false;

    for (const auto& bag_path : input_bags) {
        RosBagReader reader;
        if (!reader.open(bag_path)) {
            CVLog::Warning("[RosBagWriter] skip unreadable bag: %s",
                           bag_path.c_str());
            continue;
        }

        std::map<std::string, uint32_t> local_topic_conn;
        for (const auto& [id, conn] : reader.getConnections()) {
            (void)id;
            const uint32_t cid = writer.addConnection(conn);
            local_topic_conn[conn.topic] = cid;
        }

        reader.readMessages([&](const BagMessage& msg) {
            auto it = local_topic_conn.find(msg.topic);
            if (it == local_topic_conn.end()) return true;
            writer.writeMessage(it->second, msg.timestamp_ns, msg.data);
            return true;
        });
    }

    return writer.close();
}

bool filterRosBagByTime(const std::string& input_bag,
                        const std::string& output_bag,
                        uint64_t start_ns,
                        uint64_t end_ns) {
    RosBagReader reader;
    if (!reader.open(input_bag)) return false;

    RosBagWriter writer;
    if (!writer.open(output_bag)) return false;

    std::map<std::string, uint32_t> topic_conn;
    for (const auto& [id, conn] : reader.getConnections()) {
        (void)id;
        topic_conn[conn.topic] = writer.addConnection(conn);
    }

    size_t written = 0;
    reader.readMessages(
            [&](const BagMessage& msg) {
                auto it = topic_conn.find(msg.topic);
                if (it == topic_conn.end()) return true;
                writer.writeMessage(it->second, msg.timestamp_ns, msg.data);
                ++written;
                return true;
            },
            {}, start_ns, end_ns);

    if (!writer.close()) return false;
    CVLog::Print(
            "[RosBagWriter] filtered %s -> %s (%zu messages, %.3fs window)",
            input_bag.c_str(), output_bag.c_str(), written,
            static_cast<double>(end_ns - start_ns) / 1e9);
    return written > 0;
}

}  // namespace mcalib
