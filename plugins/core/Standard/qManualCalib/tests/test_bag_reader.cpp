// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <string>

#include "BagAlignment.h"
#include "BagDiscovery.h"
#include "CloudDecombine.h"
#include "ProtoDecoder.h"
#include "RosBagReader.h"

#ifndef MCALIB_TEST_DATA_DIR
#define MCALIB_TEST_DATA_DIR "tests/data"
#endif

static std::string testDataPath(const char* relative) {
    return std::string(MCALIB_TEST_DATA_DIR) + "/" + relative;
}

static const std::string TEST_BAG = testDataPath("bags/sample_aligned.bag");

static int g_fail = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "  FAIL: " << msg << " (" #cond ")" << std::endl; \
            ++g_fail;                                                      \
            return;                                                        \
        }                                                                  \
    } while (0)

static void test_open_and_duration() {
    std::cout << "=== test_open_and_duration ===" << std::endl;
    mcalib::RosBagReader reader;
    bool ok = reader.open(TEST_BAG);
    std::cout << "  open() returned: " << (ok ? "true" : "false") << std::endl;
    CHECK(ok, "Failed to open bag file");

    uint64_t dur = reader.getDuration();
    double dur_s = dur / 1e9;
    std::cout << "  duration_ns = " << dur << std::endl;
    std::cout << "  duration_s  = " << dur_s << std::endl;
    CHECK(dur_s >= 0.2, "Duration too short");
    CHECK(dur_s < 86400.0, "Duration too long (>24h)");
    std::cout << "  PASS: duration is reasonable (" << dur_s << " s)"
              << std::endl;

    uint64_t begin = reader.getBeginTime();
    uint64_t end = reader.getEndTime();
    std::cout << "  begin_time_ns = " << begin << std::endl;
    std::cout << "  end_time_ns   = " << end << std::endl;
    CHECK(end > begin, "end <= begin");

    double begin_epoch_s = begin / 1e9;
    std::cout << "  begin_epoch_s = " << begin_epoch_s << std::endl;
    CHECK(begin_epoch_s > 1e9 && begin_epoch_s < 2e9,
          "begin timestamp not a plausible Unix epoch");
    std::cout << "  PASS: timestamps are plausible Unix epoch" << std::endl;
}

static void test_topic_listing() {
    std::cout << "\n=== test_topic_listing ===" << std::endl;
    mcalib::RosBagReader reader;
    CHECK(reader.open(TEST_BAG), "Failed to open bag");

    auto types = reader.getTopicTypes();
    std::cout << "  total topics: " << types.size() << std::endl;
    CHECK(types.size() > 10, "Too few topics found");

    int cam = 0, lidar = 0;
    for (const auto& [topic, type] : types) {
        if (topic.find("/sensors/camera/") != std::string::npos &&
            topic.find("compressed_proto") != std::string::npos)
            ++cam;
        if (topic.find("combined_point_cloud") != std::string::npos) ++lidar;
    }
    std::cout << "  camera topics: " << cam << ", lidar topics: " << lidar
              << std::endl;
    CHECK(cam > 0, "No camera topics found");
    std::cout << "  PASS" << std::endl;
}

static void test_read_camera_message_perf() {
    std::cout << "\n=== test_read_camera_message_perf ===" << std::endl;
    mcalib::RosBagReader reader;
    CHECK(reader.open(TEST_BAG), "Failed to open bag");

    std::string topic = "/sensors/camera/camera_1_raw_data/compressed_proto";

    auto t0 = std::chrono::steady_clock::now();
    auto msg = reader.readMessageAtPercent(topic, 0.5);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  readMessageAtPercent(50%) took " << ms << " ms"
              << std::endl;
    std::cout << "  msg.data.size() = " << msg.data.size() << std::endl;
    std::cout << "  msg.timestamp_ns = " << msg.timestamp_ns << std::endl;
    CHECK(!msg.data.empty(), "No message data found at 50%");
    CHECK(ms < 5000.0, "Read took too long (>5s)");
    std::cout << "  PASS" << std::endl;
}

static void test_read_with_time_filter() {
    std::cout << "\n=== test_read_with_time_filter ===" << std::endl;
    mcalib::RosBagReader reader;
    CHECK(reader.open(TEST_BAG), "Failed to open bag");

    uint64_t begin = reader.getBeginTime();
    uint64_t dur = reader.getDuration();
    uint64_t mid = begin + dur / 2;
    uint64_t window = 500000000ULL;

    std::set<std::string> topics = {
            "/sensors/camera/camera_1_raw_data/compressed_proto"};

    int count = 0;
    auto t0 = std::chrono::steady_clock::now();
    reader.readMessages(
            [&](const mcalib::BagMessage& msg) {
                ++count;
                return true;
            },
            topics, mid - window, mid + window);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  messages in 1s window at midpoint: " << count << std::endl;
    std::cout << "  took " << ms << " ms" << std::endl;
    CHECK(count > 0, "No messages found in time window");
    CHECK(ms < 3000.0, "Read took too long (>3s)");
    std::cout << "  PASS" << std::endl;
}

static void test_multiple_reads_perf() {
    std::cout << "\n=== test_multiple_reads_perf ===" << std::endl;
    mcalib::RosBagReader reader;
    CHECK(reader.open(TEST_BAG), "Failed to open bag");

    std::vector<std::string> cam_topics = {
            "/sensors/camera/camera_1_raw_data/compressed_proto",
            "/sensors/camera/camera_2_raw_data/compressed_proto",
            "/sensors/camera/camera_3_raw_data/compressed_proto"};

    auto t0 = std::chrono::steady_clock::now();
    for (const auto& topic : cam_topics) {
        auto msg = reader.readMessageAtPercent(topic, 0.25);
        std::cout << "  " << topic << ": " << msg.data.size() << " bytes"
                  << std::endl;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  3 camera reads at 25% took " << ms << " ms total"
              << std::endl;
    CHECK(ms < 10000.0, "3 reads took too long (>10s)");
    std::cout << "  PASS" << std::endl;
}

static void test_church_header_strip() {
    std::cout << "\n=== test_church_header_strip ===" << std::endl;

    std::string no_header = "hello";
    std::string stripped = mcalib::ProtoDecoder::stripChurchHeader(no_header);
    CHECK(stripped == "hello",
          "stripChurchHeader should return unchanged data without $$$$");

    std::string with_header;
    with_header += "$$$$";
    uint32_t hlen = 10;
    with_header.append(reinterpret_cast<const char*>(&hlen), 4);
    with_header += "0123456789";
    with_header += "PAYLOAD";
    stripped = mcalib::ProtoDecoder::stripChurchHeader(with_header);
    CHECK(stripped == "PAYLOAD",
          "stripChurchHeader should return PAYLOAD after church header");
    std::cout << "  PASS: church header stripping works" << std::endl;
}

static void test_proto_decode_image() {
    std::cout << "\n=== test_proto_decode_image ===" << std::endl;
    mcalib::RosBagReader reader;
    CHECK(reader.open(TEST_BAG), "Failed to open bag");

    std::string topic = "/sensors/camera/camera_1_raw_data/compressed_proto";
    auto msg = reader.readFirstMessage(topic);
    CHECK(!msg.data.empty(), "No message data");

    std::cout << "  raw msg size: " << msg.data.size() << " bytes" << std::endl;

    // Check for church header
    if (msg.data.size() >= 8) {
        uint32_t str_len;
        std::memcpy(&str_len, msg.data.data(), 4);
        std::string inner = msg.data.substr(4, str_len);
        bool has_church =
                (inner.size() >= 4 && inner.compare(0, 4, "$$$$") == 0);
        std::cout << "  has church header: " << (has_church ? "YES" : "NO")
                  << std::endl;
        if (has_church) {
            uint32_t hlen;
            std::memcpy(&hlen, inner.data() + 4, 4);
            std::cout << "  church header_len: " << hlen << std::endl;
            std::cout << "  actual proto payload: " << (inner.size() - 8 - hlen)
                      << " bytes" << std::endl;
        }
    }

    cv::Mat image;
    double ts = 0;
    bool ok = mcalib::ProtoDecoder::decodeCompressedImageFromBag(msg.data,
                                                                 image, ts);
    std::cout << "  decode result: " << (ok ? "SUCCESS" : "FAILED")
              << std::endl;
    CHECK(ok, "Failed to decode CompressedImage from bag");

    std::cout << "  image size: " << image.cols << "x" << image.rows
              << " channels=" << image.channels() << std::endl;
    std::cout << "  timestamp: " << ts << std::endl;
    CHECK(image.cols > 0 && image.rows > 0,
          "Decoded image has zero dimensions");
    CHECK(ts > 0, "Timestamp should be positive");
    std::cout << "  PASS: image decoded successfully!" << std::endl;
}

static void test_proto_decode_pointcloud() {
    std::cout << "\n=== test_proto_decode_pointcloud ===" << std::endl;
    mcalib::RosBagReader reader;
    CHECK(reader.open(TEST_BAG), "Failed to open bag");

    // Find the actual lidar topic
    std::string topic;
    auto types = reader.getTopicTypes();
    for (const auto& [t, tp] : types) {
        if (t.find("combined_point_cloud") != std::string::npos) {
            topic = t;
            break;
        }
    }
    std::cout << "  lidar topic: " << (topic.empty() ? "NOT FOUND" : topic)
              << std::endl;
    CHECK(!topic.empty(), "No combined_point_cloud topic found");
    auto msg = reader.readFirstMessage(topic);
    CHECK(!msg.data.empty(), "No point cloud message data");

    std::cout << "  raw msg size: " << msg.data.size() << " bytes" << std::endl;

    mcalib::ProtoDecoder::PointCloud2Data cloud;
    bool ok = mcalib::ProtoDecoder::decodePointCloud2FromBag(msg.data, cloud);
    std::cout << "  decode result: " << (ok ? "SUCCESS" : "FAILED")
              << std::endl;
    CHECK(ok, "Failed to decode PointCloud2 from bag");

    std::cout << "  width=" << cloud.width << " height=" << cloud.height
              << " point_step=" << cloud.point_step
              << " data_size=" << cloud.data.size() << std::endl;
    CHECK(cloud.width > 0, "Point cloud has zero width");
    CHECK(!cloud.data.empty(), "Point cloud data is empty");

    size_t num_points = cloud.data.size() / cloud.point_step;
    std::cout << "  estimated points: " << num_points << std::endl;
    std::cout << "  frame_id=" << cloud.frame_id
              << " embedded_lidars=" << cloud.embedded_lidars.size()
              << std::endl;

    std::vector<mcalib::PointXYZIRT> cloud_raw;
    std::string frame_id;
    int64_t stamp_us = 0;
    const bool decombined = mcalib::decombinePointCloud(
            cloud, nullptr, cloud_raw, frame_id, stamp_us);
    std::cout << "  decombine (embedded only): "
              << (decombined ? "OK" : "SKIP/FAIL")
              << " points=" << cloud_raw.size() << " frame=" << frame_id
              << std::endl;
    if (!cloud.embedded_lidars.empty()) {
        CHECK(decombined, "embedded lidar_configs decombine failed");
        CHECK(frame_id == "lidar_uncalibrated", "decombined frame_id mismatch");
    }

    std::cout << "  PASS: point cloud decoded successfully!" << std::endl;
}

static void test_bev_blend_weights() {
    std::cout << "\n=== test_bev_blend_weights ===" << std::endl;

    cv::Size bev_size(100, 100);

    cv::Mat mask1 = cv::Mat::zeros(bev_size, CV_8UC1);
    cv::rectangle(mask1, cv::Rect(10, 10, 80, 50), cv::Scalar(255), -1);

    cv::Mat mask2 = cv::Mat::zeros(bev_size, CV_8UC1);
    cv::rectangle(mask2, cv::Rect(10, 40, 80, 50), cv::Scalar(255), -1);

    cv::Mat overlap = mask1 & mask2;
    cv::Mat overlap_inv = ~overlap;
    cv::Mat single1 = mask1 & overlap_inv;
    cv::Mat single2 = mask2 & overlap_inv;

    cv::Mat dist1, dist2;
    cv::distanceTransform(255 - single1, dist1, cv::DIST_L2, 3);
    cv::distanceTransform(255 - single2, dist2, cv::DIST_L2, 3);

    const int overlap_y = 45;
    const float d1 = dist1.at<float>(overlap_y, 50);
    const float d2 = dist2.at<float>(overlap_y, 50);
    const float total = d1 + d2;
    const float w1 = (total > 1e-6f) ? d2 / total : 0.f;
    const float w2 = (total > 1e-6f) ? d1 / total : 0.f;

    std::cout << "  overlap row d1=" << d1 << " d2=" << d2 << " w1=" << w1
              << " w2=" << w2 << std::endl;
    CHECK(w1 > 0.1f && w2 > 0.1f, "Overlap blend weights should be shared");
    CHECK(std::fabs(w1 + w2 - 1.0f) < 0.01f, "Overlap weights should sum to 1");

    std::cout << "  PASS: blend weights follow codetree distanceTransform"
              << std::endl;
}

static void test_bev_group_sync() {
    std::cout << "\n=== test_bev_group_sync ===" << std::endl;
    mcalib::RosBagReader reader;
    CHECK(reader.open(TEST_BAG), "Failed to open bag");

    std::vector<std::string> svm_topics;
    std::vector<std::string> avm_topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("/sensors/camera/") == std::string::npos ||
            topic.find("compressed_proto") == std::string::npos) {
            continue;
        }
        if (topic.find("panoramic_") != std::string::npos) {
            avm_topics.push_back(topic);
        } else {
            svm_topics.push_back(topic);
        }
    }
    std::sort(svm_topics.begin(), svm_topics.end());
    std::sort(avm_topics.begin(), avm_topics.end());
    std::cout << "  svm topics: " << svm_topics.size() << std::endl;
    std::cout << "  avm topics: " << avm_topics.size() << std::endl;
    CHECK(svm_topics.size() == 7, "Expected 7 SVM camera topics");
    CHECK(avm_topics.size() == 4, "Expected 4 AVM panoramic topics");

    int pan_msg_count = 0;
    std::set<std::string> pan_set(avm_topics.begin(), avm_topics.end());
    reader.readMessages(
            [&](const mcalib::BagMessage& msg) {
                ++pan_msg_count;
                return true;
            },
            pan_set, reader.getBeginTime(), reader.getEndTime());
    std::cout << "  panoramic messages in bag: " << pan_msg_count << std::endl;
    CHECK(pan_msg_count > 0, "No panoramic messages in sliced bag");

    for (double percent : {0.1, 0.5, 0.9}) {
        std::map<std::string, cv::Mat> images;
        std::map<std::string, int64_t> stamps_ns;
        const bool ok = mcalib::getAlignedImagesBevGroups(
                reader, svm_topics, avm_topics, percent, images, stamps_ns,
                nullptr);
        std::cout << "  bev groups @" << (percent * 100.0)
                  << "%: " << (ok ? "OK" : "FAIL")
                  << " images=" << images.size() << std::endl;
        CHECK(ok, "BEV group sync failed");
        CHECK(images.size() == 11, "Expected 7 SVM + 4 AVM synced images");
    }

    int64_t prev_center = 0;
    for (double percent : {0.1, 0.5, 0.9}) {
        std::map<std::string, cv::Mat> images;
        std::map<std::string, int64_t> stamps_ns;
        CHECK(mcalib::getAlignedImagesBevGroups(reader, svm_topics, avm_topics,
                                                percent, images, stamps_ns,
                                                nullptr),
              "BEV distinct-frame check failed");
        int64_t center = 0;
        int count = 0;
        for (const auto& [_, stamp] : stamps_ns) {
            center += stamp;
            ++count;
        }
        if (count > 0) center /= count;
        std::cout << "  bev center stamp @" << (percent * 100.0)
                  << "%: " << center << std::endl;
        if (prev_center > 0) {
            CHECK(std::llabs(center - prev_center) > 50000000LL,
                  "BEV sync groups should differ across slider positions");
        }
        prev_center = center;
    }
    std::cout << "  PASS" << std::endl;
}

static void test_lidar_group_cloud_sync() {
    std::cout << "\n=== test_lidar_group_cloud_sync ===" << std::endl;
    mcalib::RosBagReader reader;
    CHECK(reader.open(TEST_BAG), "Failed to open bag");

    std::vector<std::string> svm_topics;
    std::vector<std::string> avm_topics;
    std::vector<std::string> cloud_topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("combined_point_cloud") != std::string::npos) {
            cloud_topics.push_back(topic);
            continue;
        }
        if (topic.find("/sensors/camera/") == std::string::npos ||
            topic.find("compressed_proto") == std::string::npos) {
            continue;
        }
        if (topic.find("panoramic_") != std::string::npos) {
            avm_topics.push_back(topic);
        } else {
            svm_topics.push_back(topic);
        }
    }
    CHECK(!cloud_topics.empty(), "No lidar topic in bag");
    CHECK(!svm_topics.empty() && !avm_topics.empty(), "Missing camera topics");

    double svm_percent = 0.5;
    int64_t svm_ref = 0;
    CHECK(mcalib::findBestAlignedPercent(reader, svm_topics, 0.0, 1.0,
                                         svm_percent, svm_ref),
          "SVM sync search failed");
    double avm_percent = 0.5;
    int64_t avm_ref = 0;
    CHECK(mcalib::findBestAlignedPercent(reader, avm_topics, 0.0, 1.0,
                                         avm_percent, avm_ref),
          "AVM sync search failed");

    int svm_near = 0;
    int avm_near = 0;
    std::set<std::string> cloud_set(cloud_topics.begin(), cloud_topics.end());
    reader.readMessages(
            [&](const mcalib::BagMessage& msg) {
                double ts = 0;
                if (!mcalib::ProtoDecoder::extractPointCloudTimestampFromBag(
                            msg.data, ts)) {
                    return true;
                }
                const int64_t proto = static_cast<int64_t>(ts * 1e9);
                if (std::llabs(proto - svm_ref) < 100000000LL) ++svm_near;
                if (std::llabs(proto - avm_ref) < 100000000LL) ++avm_near;
                return true;
            },
            cloud_set, reader.getBeginTime(), reader.getEndTime());
    std::cout << "  svm_ref_clouds=" << svm_near
              << " avm_ref_clouds=" << avm_near << std::endl;
    CHECK(svm_near > 0, "No SVM-aligned clouds in sliced bag");
    CHECK(avm_near > 0, "No AVM-aligned clouds in sliced bag");

    const double percent = 0.5;
    for (const auto& [label, topics] :
         std::vector<std::pair<std::string, std::vector<std::string>>>{
                 {"svm", svm_topics}, {"avm", avm_topics}}) {
        std::map<std::string, cv::Mat> images;
        std::map<std::string, int64_t> stamps_ns;
        std::vector<mcalib::PointXYZIRT> cloud_raw;
        int64_t cloud_stamp_us = 0;
        const bool ok = mcalib::getAlignedImagesCloud(
                reader, topics, cloud_topics, percent, false, images, stamps_ns,
                cloud_raw, cloud_stamp_us, nullptr, nullptr);
        std::cout << "  " << label << " cloud @" << (percent * 100.0)
                  << "%: " << (ok ? "OK" : "FAIL")
                  << " images=" << images.size()
                  << " cloud=" << cloud_raw.size() << std::endl;
        CHECK(ok, (label + " image+cloud sync failed").c_str());
        CHECK(!cloud_raw.empty(), (label + " cloud empty").c_str());
    }
    std::cout << "  PASS" << std::endl;
}

static void test_bev_proto_sync_long_bag() {
    const char* long_bag =
            "/home/ludahai/develop/data/robotaxi_data/"
            "YR-EC15S-29_20260624_025519/bags/merge.bag";
    if (FILE* f = std::fopen(long_bag, "rb")) {
        std::fclose(f);
    } else {
        std::cout << "\n=== test_bev_proto_sync_long_bag ===" << std::endl;
        std::cout << "  SKIP: long bag not found" << std::endl;
        return;
    }

    std::cout << "\n=== test_bev_proto_sync_long_bag ===" << std::endl;
    mcalib::RosBagReader reader;
    CHECK(reader.open(long_bag), "Failed to open long bag");

    std::vector<std::string> svm_topics;
    std::vector<std::string> avm_topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("/sensors/camera/") == std::string::npos ||
            topic.find("compressed_proto") == std::string::npos) {
            continue;
        }
        if (topic.find("panoramic_") != std::string::npos) {
            avm_topics.push_back(topic);
        } else {
            svm_topics.push_back(topic);
        }
    }
    std::sort(svm_topics.begin(), svm_topics.end());
    std::sort(avm_topics.begin(), avm_topics.end());
    CHECK(svm_topics.size() >= 6, "Expected SVM topics in long bag");
    CHECK(avm_topics.size() == 4, "Expected 4 AVM topics in long bag");

    for (double percent : {0.25, 0.5, 0.75}) {
        for (const auto& [label, topics] :
             std::vector<std::pair<std::string, std::vector<std::string>>>{
                     {"svm", svm_topics}, {"avm", avm_topics}}) {
            std::map<std::string, cv::Mat> images;
            std::map<std::string, int64_t> stamps_ns;
            const bool ok = mcalib::getAlignedImages(
                    reader, topics, percent, images, stamps_ns, nullptr);
            std::cout << "  " << label << " @" << (percent * 100.0)
                      << "%: " << (ok ? "OK" : "FAIL")
                      << " images=" << images.size() << "/" << topics.size()
                      << std::endl;
            CHECK(ok, (label + " proto sync failed on long bag").c_str());
            CHECK(images.size() == topics.size(),
                  (label + " partial image set on long bag").c_str());
        }
    }
    std::cout << "  PASS" << std::endl;
}

static void test_bag_discovery_topic_group_key() {
    std::cout << "\n=== test_bag_discovery_topic_group_key ===" << std::endl;
    CHECK(mcalib::extractTopicGroupSessionKey(
                  "YR-VF6-1_20260717_040330.Heavy_Topic_Group") ==
                  "YR-VF6-1_20260717_040330",
          "Heavy topic-group session key mismatch");
    CHECK(mcalib::extractTopicGroupName(
                  "YR-VF6-1_20260717_040330.Medium_Topic_Group") ==
                  "Medium_Topic_Group",
          "Medium topic-group name mismatch");
    CHECK(mcalib::isIncludedTopicGroupBag(
                  "YR-B26A1-1_20251117_031300.Medium_Topic_Group.bag"),
          "Medium bag should be included");
    CHECK(!mcalib::isIncludedTopicGroupBag(
                  "YR-B26A1-1_20251117_031300.Tiny_Topic_Group"),
          "Tiny bag should be excluded");
    CHECK(mcalib::extractTopicGroupSessionKey("merge") == "",
          "Non topic-group name should not match");
    CHECK(mcalib::isTopicGroupDirectoryName("Heavy_Topic_Group"),
          "Heavy_Topic_Group dir name");
    CHECK(!mcalib::isTopicGroupDirectoryName("raw_bags"),
          "raw_bags is not topic-group dir");
    std::cout << "  PASS" << std::endl;
}

static void test_bag_discovery_real_layouts() {
    std::cout << "\n=== test_bag_discovery_real_layouts ===" << std::endl;

    const char* flat_root =
            "/home/ludahai/develop/data/eol/YR_VF6_1_online/bags/orig";
    const char* nested_root =
            "/home/ludahai/develop/data/eol/B26A_online/"
            "YR-B26A1-1_20251117_031232_lidar/bags/unimportant";

    if (FILE* f = std::fopen(flat_root, "rb")) {
        std::fclose(f);
        const auto flat = mcalib::discoverBagLayout(flat_root);
        CHECK(flat.layout == mcalib::BagLayoutType::FlatTopicGroup,
              "Flat topic-group layout expected");
        CHECK(flat.sessions.size() >= 2, "Expected multiple flat sessions");
        for (const auto& session : flat.sessions) {
            CHECK(!session.bag_paths.empty(), "Flat session has no bags");
        }
        const auto resolved = mcalib::resolveBagInput(flat_root, 0);
        CHECK(resolved.ok, "Flat session resolve failed");
        CHECK(resolved.source_bags.size() >= 2,
              "Flat session should include multiple topic-group bags");
        std::cout << "  flat sessions: " << flat.sessions.size()
                  << ", first session bags: " << resolved.source_bags.size()
                  << std::endl;
    } else {
        std::cout << "  SKIP flat layout: path not found" << std::endl;
    }

    if (FILE* f = std::fopen(nested_root, "rb")) {
        std::fclose(f);
        const auto nested = mcalib::discoverBagLayout(nested_root);
        CHECK(nested.layout == mcalib::BagLayoutType::NestedTopicGroup,
              "Nested topic-group layout expected");
        CHECK(!nested.sessions.empty(), "Expected nested sessions");
        bool found_multi_group = false;
        for (const auto& session : nested.sessions) {
            const auto bags =
                    mcalib::filterIncludedSessionBags(session.bag_paths);
            if (bags.size() >= 3) {
                found_multi_group = true;
            }
        }
        CHECK(found_multi_group,
              "Expected nested session with Heavy/Light/Medium bags");
        std::cout << "  nested sessions: " << nested.sessions.size()
                  << std::endl;
    } else {
        std::cout << "  SKIP nested layout: path not found" << std::endl;
    }

    const auto single = mcalib::discoverBagLayout(TEST_BAG);
    CHECK(single.layout == mcalib::BagLayoutType::SingleFile,
          "Sample aligned bag should be single-file layout");
    CHECK(single.sessions.size() == 1, "Single bag should expose one session");
    std::cout << "  PASS" << std::endl;
}

static void test_open_multi_topic_group() {
    std::cout << "\n=== test_open_multi_topic_group ===" << std::endl;
    const char* flat_root =
            "/home/ludahai/develop/data/eol/YR_VF6_1_online/bags/orig";
    if (FILE* f = std::fopen(flat_root, "rb")) {
        std::fclose(f);
    } else {
        std::cout << "  SKIP: flat layout path not found" << std::endl;
        return;
    }

    const auto resolved = mcalib::resolveBagInput(flat_root, 0);
    CHECK(resolved.ok, "resolveBagInput failed");
    CHECK(resolved.source_bags.size() >= 3,
          "Expected Heavy/Light/Medium bags in first session");

    mcalib::RosBagReader reader;
    CHECK(reader.openMulti(resolved.source_bags), "openMulti failed");
    CHECK(reader.isMultiBagMode(), "Expected multi-bag mode");

    auto topics = reader.getTopics();
    bool has_camera = false;
    bool has_pose = false;
    bool has_cloud = false;
    for (const auto& topic : topics) {
        if (topic.find("/sensors/camera/") != std::string::npos) {
            has_camera = true;
        }
        if (topic.find("/localization/pose") != std::string::npos) {
            has_pose = true;
        }
        if (topic.find("point_cloud") != std::string::npos ||
            topic.find("lidar") != std::string::npos) {
            has_cloud = true;
        }
    }
    std::cout << "  topics=" << topics.size() << " camera=" << has_camera
              << " pose=" << has_pose << " cloud=" << has_cloud << std::endl;
    CHECK(has_camera, "Missing camera topics from Heavy bag");
    CHECK(has_pose, "Missing pose topics from Light bag");
    CHECK(has_cloud, "Missing lidar/cloud topics from Medium bag");

    const std::set<std::string> cam_topics = {
            "/sensors/camera/camera_1_raw_data/compressed_proto"};
    const auto msgs = reader.readAllMessagesParallel(
            cam_topics, nullptr, reader.getBeginTime(),
            reader.getBeginTime() + 500000000ULL);
    CHECK(!msgs.empty(),
          "readAllMessagesParallel returned no camera msgs in multi-bag mode");
    std::cout << "  parallel camera msgs=" << msgs.size() << std::endl;
    std::cout << "  PASS" << std::endl;
}

static void test_merged_single_bag_file() {
    std::cout << "\n=== test_merged_single_bag_file ===" << std::endl;
    const auto resolved = mcalib::discoverBagLayout(TEST_BAG);
    CHECK(resolved.layout == mcalib::BagLayoutType::SingleFile,
          "sample_aligned.bag should be merged single-file layout");
    CHECK(resolved.sessions.size() == 1, "Expected one session");

    const auto opened = mcalib::resolveBagInput(TEST_BAG, 0);
    CHECK(opened.ok, "resolveBagInput failed for merged bag");
    CHECK(opened.source_bags.size() == 1,
          "Merged bag should resolve to one file");
    CHECK(!opened.readable_path.empty(),
          "Merged bag should expose readable path");

    mcalib::RosBagReader reader;
    CHECK(reader.open(opened.source_bags.front()), "open merged bag failed");
    CHECK(!reader.isMultiBagMode(), "Merged bag must not use openMulti");

    auto topics = reader.getTopicTypes();
    bool has_camera = false;
    bool has_pose = false;
    bool has_cloud = false;
    for (const auto& [topic, _] : topics) {
        if (topic.find("/sensors/camera/") != std::string::npos)
            has_camera = true;
        if (topic.find("/localization/pose") != std::string::npos)
            has_pose = true;
        if (topic.find("point_cloud") != std::string::npos ||
            topic.find("lidar") != std::string::npos) {
            has_cloud = true;
        }
    }
    std::cout << "  topics=" << topics.size() << " camera=" << has_camera
              << " pose=" << has_pose << " cloud=" << has_cloud << std::endl;
    CHECK(has_camera, "Merged bag missing camera topics");
    CHECK(has_pose, "Merged bag missing pose topics");
    CHECK(has_cloud, "Merged bag missing cloud topics");
    std::cout << "  PASS" << std::endl;
}

static void test_yr_vf6_hevc_multi_bag() {
    const char* flat_dir =
            "/home/ludahai/develop/data/eol/YR_VF6_1_online/bags/orig";
    if (FILE* f = std::fopen(flat_dir, "rb")) {
        std::fclose(f);
    } else {
        std::cout << "\n=== test_yr_vf6_hevc_multi_bag ===" << std::endl;
        std::cout << "  SKIP: YR_VF6 flat dir not found" << std::endl;
        return;
    }

    std::cout << "\n=== test_yr_vf6_hevc_multi_bag ===" << std::endl;
    const auto resolved = mcalib::resolveBagInput(flat_dir, 0);
    CHECK(resolved.ok, "resolveBagInput failed for YR_VF6 flat dir");
    CHECK(resolved.source_bags.size() >= 2,
          "Expected multi-bag topic group for YR_VF6");

    mcalib::RosBagReader reader;
    CHECK(reader.openMulti(resolved.source_bags),
          "openMulti failed for YR_VF6");

    std::vector<std::string> image_topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("/sensors/camera/") != std::string::npos &&
            topic.find("compressed_proto") != std::string::npos) {
            image_topics.push_back(topic);
        }
    }
    std::sort(image_topics.begin(), image_topics.end());
    CHECK(image_topics.size() >= 4,
          "Expected camera topics in YR_VF6 heavy bag");

    std::map<std::string, cv::Mat> images;
    std::map<std::string, int64_t> stamps_ns;
    const bool ok = mcalib::getAlignedImages(reader, image_topics, 0.5, images,
                                             stamps_ns, nullptr);
    std::cout << "  getAlignedImages @50%: " << (ok ? "OK" : "FAIL")
              << " images=" << images.size() << "/" << image_topics.size()
              << std::endl;
    CHECK(ok, "YR_VF6 HEVC image sync failed");
    CHECK(images.size() == image_topics.size(), "YR_VF6 partial image set");
    for (const auto& [topic, image] : images) {
        (void)topic;
        CHECK(!image.empty(), "Decoded YR_VF6 image is empty");
        CHECK(image.cols > 0 && image.rows > 0, "Invalid YR_VF6 image size");
    }
    std::cout << "  PASS" << std::endl;
}

static void test_yr_vf6_hevc_merge_bag() {
    const char* merge_dir =
            "/home/ludahai/develop/data/eol/YR_VF6_1_online/bags";
    if (FILE* f = std::fopen(merge_dir, "rb")) {
        std::fclose(f);
    } else {
        std::cout << "\n=== test_yr_vf6_hevc_merge_bag ===" << std::endl;
        std::cout << "  SKIP: YR_VF6 merge dir not found" << std::endl;
        return;
    }

    std::cout << "\n=== test_yr_vf6_hevc_merge_bag ===" << std::endl;
    const auto resolved = mcalib::resolveBagInput(merge_dir, 0);
    CHECK(resolved.ok, "resolveBagInput failed for YR_VF6 merge dir");
    CHECK(resolved.layout == mcalib::BagLayoutType::SingleFile,
          "Expected merge.bag single-file layout");

    mcalib::RosBagReader reader;
    CHECK(reader.open(resolved.readable_path), "open merge.bag failed");

    std::vector<std::string> image_topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("/sensors/camera/") != std::string::npos &&
            topic.find("compressed_proto") != std::string::npos) {
            image_topics.push_back(topic);
        }
    }
    std::sort(image_topics.begin(), image_topics.end());
    image_topics.resize(std::min(image_topics.size(), size_t(4)));
    CHECK(!image_topics.empty(), "Expected camera topics in merge.bag");

    std::map<std::string, cv::Mat> images;
    std::map<std::string, int64_t> stamps_ns;
    const bool ok = mcalib::getAlignedImages(reader, image_topics, 0.5, images,
                                             stamps_ns, nullptr);
    std::cout << "  getAlignedImages @50%: " << (ok ? "OK" : "FAIL")
              << " images=" << images.size() << "/" << image_topics.size()
              << std::endl;
    CHECK(ok, "YR_VF6 merge.bag HEVC image sync failed");
    for (const auto& [topic, image] : images) {
        std::cout << "  " << topic << " -> " << image.cols << "x" << image.rows
                  << std::endl;
        (void)topic;
        CHECK(!image.empty(), "Decoded merge.bag image is empty");
    }
    std::cout << "  PASS" << std::endl;
}

static void test_yr_vf6_hevc_multi_bag_scrub() {
    const char* flat_dir =
            "/home/ludahai/develop/data/eol/YR_VF6_1_online/bags/orig";
    if (FILE* f = std::fopen(flat_dir, "rb")) {
        std::fclose(f);
    } else {
        std::cout << "\n=== test_yr_vf6_hevc_multi_bag_scrub ===" << std::endl;
        std::cout << "  SKIP: YR_VF6 flat dir not found" << std::endl;
        return;
    }

    std::cout << "\n=== test_yr_vf6_hevc_multi_bag_scrub ===" << std::endl;
    const auto resolved = mcalib::resolveBagInput(flat_dir, 0);
    CHECK(resolved.ok, "resolveBagInput failed for YR_VF6 flat dir");

    mcalib::RosBagReader reader;
    CHECK(reader.openMulti(resolved.source_bags),
          "openMulti failed for YR_VF6");

    std::vector<std::string> image_topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("/sensors/camera/") != std::string::npos &&
            topic.find("compressed_proto") != std::string::npos) {
            image_topics.push_back(topic);
        }
    }
    std::sort(image_topics.begin(), image_topics.end());
    CHECK(image_topics.size() >= 4, "Expected camera topics in YR_VF6 bags");

    std::set<std::string> topic_set(image_topics.begin(), image_topics.end());
    CHECK(reader.buildTopicTimeIndex(topic_set), "buildTopicTimeIndex failed");

    int ok_frames = 0;
    int total_frames = 0;
    for (int step = 0; step <= 10; ++step) {
        const double percent = step / 10.0;
        std::map<std::string, cv::Mat> images;
        std::map<std::string, int64_t> stamps_ns;
        const bool ok = mcalib::getAlignedImages(reader, image_topics, percent,
                                                 images, stamps_ns, nullptr);
        ++total_frames;
        if (ok && images.size() == image_topics.size()) {
            bool all_valid = true;
            for (const auto& [topic, image] : images) {
                (void)topic;
                if (image.empty() || image.cols < 640 || image.rows < 480) {
                    all_valid = false;
                    break;
                }
            }
            if (all_valid) ++ok_frames;
        }
        std::cout << "  scrub @" << (percent * 100.0)
                  << "%: " << (ok ? "sync" : "fail")
                  << " images=" << images.size() << "/" << image_topics.size()
                  << std::endl;
    }

    std::cout << "  valid frames: " << ok_frames << "/" << total_frames
              << std::endl;
    CHECK(ok_frames >= total_frames - 1,
          "Too many invalid HEVC scrub frames (need sequential GOP decode)");
    std::cout << "  PASS" << std::endl;

    // Performance benchmark: measure per-frame latency for slider scrubbing.
    std::cout << "\n  perf benchmark (10 frames):" << std::endl;
    double total_ms = 0;
    int bench_count = 0;
    for (int step = 0; step <= 10; ++step) {
        const double percent = step / 10.0;
        auto t0 = std::chrono::steady_clock::now();
        std::map<std::string, cv::Mat> images;
        std::map<std::string, int64_t> stamps_ns;
        mcalib::getAlignedImages(reader, image_topics, percent, images,
                                 stamps_ns, nullptr);
        auto t1 = std::chrono::steady_clock::now();
        const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (step > 0) {
            total_ms += ms;
            ++bench_count;
        }
        std::cout << "    @" << (percent * 100.0) << "%: " << ms << " ms"
                  << std::endl;
    }
    if (bench_count > 0) {
        const double avg = total_ms / bench_count;
        std::cout << "  avg latency: " << avg << " ms/frame (" << 1000.0 / avg
                  << " fps)" << std::endl;
    }
}

static void test_yr_vf6_2_hevc_11cam_bev_groups() {
    const char* bag_dir =
            "/home/ludahai/develop/data/eol/YR_VF6_2_online/YR-VF6-2_20260716_"
            "160657/bags/unimportant";
    if (FILE* f = std::fopen(bag_dir, "rb")) {
        std::fclose(f);
    } else {
        std::cout << "\n=== test_yr_vf6_2_hevc_11cam_bev_groups ==="
                  << std::endl;
        std::cout << "  SKIP: YR_VF6_2 bag dir not found" << std::endl;
        return;
    }

    std::cout << "\n=== test_yr_vf6_2_hevc_11cam_bev_groups ===" << std::endl;
    const auto resolved = mcalib::resolveBagInput(bag_dir, 0);
    CHECK(resolved.ok, "resolveBagInput failed for YR_VF6_2");

    mcalib::RosBagReader reader;
    CHECK(reader.openMulti(resolved.source_bags),
          "openMulti failed for YR_VF6_2");

    std::vector<std::string> svm_topics;
    std::vector<std::string> avm_topics;
    std::vector<std::string> cloud_topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("combined_point_cloud") != std::string::npos) {
            cloud_topics.push_back(topic);
            continue;
        }
        if (topic.find("/sensors/camera/") == std::string::npos ||
            topic.find("compressed_proto") == std::string::npos) {
            continue;
        }
        if (topic.find("panoramic_") != std::string::npos) {
            avm_topics.push_back(topic);
        } else {
            svm_topics.push_back(topic);
        }
    }
    std::sort(svm_topics.begin(), svm_topics.end());
    std::sort(avm_topics.begin(), avm_topics.end());
    CHECK(svm_topics.size() == 7, "Expected 7 SVM topics");
    CHECK(avm_topics.size() == 4, "Expected 4 AVM topics");
    CHECK(!cloud_topics.empty(), "Expected lidar cloud topic");

    std::set<std::string> index_topics(svm_topics.begin(), svm_topics.end());
    index_topics.insert(avm_topics.begin(), avm_topics.end());
    CHECK(reader.buildTopicTimeIndex(index_topics),
          "buildTopicTimeIndex failed");

    int ok_frames = 0;
    for (int step = 0; step <= 10; ++step) {
        const double percent = step / 10.0;
        std::vector<std::string> all_topics = svm_topics;
        all_topics.insert(all_topics.end(), avm_topics.begin(),
                          avm_topics.end());

        std::map<std::string, cv::Mat> images;
        std::map<std::string, int64_t> stamps_ns;
        int64_t cloud_stamp_ns = 0;
        const bool ok = mcalib::getAlignedImagesForBev(
                reader, all_topics, percent, images, stamps_ns, cloud_stamp_ns,
                nullptr);
        std::cout << "  bev @" << (percent * 100.0)
                  << "%: " << (ok ? "OK" : "FAIL")
                  << " images=" << images.size()
                  << "/11 cloud_ns=" << cloud_stamp_ns << std::endl;
        if (ok && images.size() == 11) {
            bool all_valid = true;
            for (const auto& [topic, image] : images) {
                (void)topic;
                if (image.empty() || image.cols < 640 || image.rows < 480) {
                    all_valid = false;
                    break;
                }
            }
            if (all_valid) ++ok_frames;

            std::vector<mcalib::PointXYZIRT> cloud_raw;
            int64_t cloud_stamp_us = 0;
            const bool cloud_ok = mcalib::getAlignedImagesCloud(
                    reader, svm_topics, cloud_topics, percent, false, images,
                    stamps_ns, cloud_raw, cloud_stamp_us, nullptr, nullptr);
            std::cout << "    svm+lidar @" << (percent * 100.0)
                      << "%: cloud=" << cloud_raw.size()
                      << (cloud_ok ? " OK" : " FAIL") << std::endl;
            CHECK(cloud_ok && !cloud_raw.empty(),
                  "SVM+lidar sync failed at scrub position");
        }
    }
    CHECK(ok_frames >= 8, "Too few valid 11-cam HEVC frames across scrub");
    std::cout << "  valid 11-cam frames: " << ok_frames << "/11 positions"
              << std::endl;
    std::cout << "  PASS" << std::endl;
}

int main() {
    std::cout << "=============================" << std::endl;
    std::cout << "RosBagReader Test Suite" << std::endl;
    std::cout << "=============================" << std::endl;

    test_open_and_duration();
    test_topic_listing();
    test_read_camera_message_perf();
    test_read_with_time_filter();
    test_multiple_reads_perf();
    test_church_header_strip();
    test_proto_decode_image();
    test_proto_decode_pointcloud();
    test_bev_group_sync();
    test_bev_proto_sync_long_bag();
    test_lidar_group_cloud_sync();
    test_bev_blend_weights();
    test_bag_discovery_topic_group_key();
    test_bag_discovery_real_layouts();
    test_open_multi_topic_group();
    test_merged_single_bag_file();
    test_yr_vf6_hevc_multi_bag();
    test_yr_vf6_hevc_multi_bag_scrub();
    test_yr_vf6_2_hevc_11cam_bev_groups();
    test_yr_vf6_hevc_merge_bag();

    std::cout << "\n=============================" << std::endl;
    if (g_fail == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << g_fail << " TEST(S) FAILED" << std::endl;
    }
    std::cout << "=============================" << std::endl;
    return g_fail;
}
