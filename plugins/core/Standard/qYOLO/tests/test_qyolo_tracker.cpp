// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Unit tests for the qYOLO multi-object tracker port (src/tracking).
// Pure algorithm tests: no AICore, no Qt — the tracker consumes plain
// detection vectors and an optional tightly-packed RGB8 frame.

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <vector>

#include "tracking/tracker.hpp"

namespace {

using qyolo::track::create_tracker;
using qyolo::track::default_tracker_config;
using qyolo::track::FrameInput;
using qyolo::track::RgbFrame;
using qyolo::track::TrackConfig;
using qyolo::track::TrackDet;
using qyolo::track::TrackedBox;

constexpr float kAxisAligned = -10.0f;

TrackDet boxAt(float x1,
               float y1,
               float x2,
               float y2,
               float score = 0.9f,
               int cls = 0,
               int idx = 0) {
    TrackDet t;
    t.cx = (x1 + x2) * 0.5f;
    t.cy = (y1 + y2) * 0.5f;
    t.w = x2 - x1;
    t.h = y2 - y1;
    t.angle = kAxisAligned;
    t.score = score;
    t.class_id = cls;
    t.idx = idx;
    return t;
}

// One uniform gray frame (valid GMC input; motion compensation sees no
// camera motion, which is exactly what a static test wants).
RgbFrame grayFrame(int w, int h, std::vector<std::uint8_t>* storage) {
    storage->assign(static_cast<size_t>(w) * static_cast<size_t>(h) * 3, 128);
    return RgbFrame{w, h, storage->data()};
}

TEST(QyoloTrackerConfig, DefaultsMirrorOfficialYamls) {
    // Values cross-checked against ultralytics/cfg/trackers/*.yaml.
    TrackConfig cfg;
    ASSERT_TRUE(default_tracker_config("bytetrack", cfg));
    EXPECT_EQ(cfg.tracker_type, "bytetrack");
    EXPECT_FLOAT_EQ(cfg.track_high_thresh, 0.25f);
    EXPECT_FLOAT_EQ(cfg.track_low_thresh, 0.1f);
    EXPECT_FLOAT_EQ(cfg.new_track_thresh, 0.25f);
    EXPECT_EQ(cfg.track_buffer, 30);
    EXPECT_FLOAT_EQ(cfg.match_thresh, 0.8f);
    EXPECT_TRUE(cfg.fuse_score);

    ASSERT_TRUE(default_tracker_config("botsort", cfg));
    EXPECT_EQ(cfg.tracker_type, "botsort");
    EXPECT_EQ(cfg.gmc_method, "sparseOptFlow");
    EXPECT_FLOAT_EQ(cfg.proximity_thresh, 0.5f);
    EXPECT_FLOAT_EQ(cfg.appearance_thresh, 0.8f);
    EXPECT_FALSE(cfg.with_reid);

    ASSERT_TRUE(default_tracker_config("ocsort", cfg));
    EXPECT_EQ(cfg.tracker_type, "ocsort");
    EXPECT_EQ(cfg.delta_t, 3);
    EXPECT_FLOAT_EQ(cfg.inertia, 0.2f);
    EXPECT_FALSE(cfg.use_byte);

    ASSERT_TRUE(default_tracker_config("deepocsort", cfg));
    EXPECT_EQ(cfg.tracker_type, "deepocsort");
    EXPECT_FLOAT_EQ(cfg.alpha_fixed_emb, 0.95f);

    ASSERT_TRUE(default_tracker_config("fasttrack", cfg));
    EXPECT_EQ(cfg.tracker_type, "fasttrack");
    EXPECT_EQ(cfg.reset_velocity_offset_occ, 5);
    EXPECT_EQ(cfg.reset_pos_offset_occ, 3);
    EXPECT_FLOAT_EQ(cfg.enlarge_bbox_occ, 1.1f);

    ASSERT_TRUE(default_tracker_config("tracktrack", cfg));
    EXPECT_EQ(cfg.tracker_type, "tracktrack");
    EXPECT_FLOAT_EQ(cfg.penalty_p, 0.2f);
    EXPECT_FLOAT_EQ(cfg.penalty_q, 0.4f);
    EXPECT_FLOAT_EQ(cfg.reduce_step, 0.05f);
    EXPECT_EQ(cfg.min_track_len, 3);

    EXPECT_FALSE(default_tracker_config("nope", cfg));
}

TEST(QyoloTrackerConfig, YamlOverride) {
    // Flat "key: value" map, same as the official tracker YAMLs; unknown
    // keys are ignored (Python getattr-with-default reads).
    const char* path = "qyolo_tracker_test_config.yaml";
    {
        std::ofstream out(path);
        out << "tracker_type: bytetrack\n";
        out << "track_buffer: 77\n";
        out << "not_a_real_key: 1\n";
    }
    TrackConfig cfg;
    EXPECT_TRUE(qyolo::track::load_tracker_config_yaml(path, cfg));
    std::remove(path);
    EXPECT_EQ(cfg.tracker_type, "bytetrack");
    EXPECT_EQ(cfg.track_buffer, 77);
    // Untouched keys keep the current values (override semantics).
    EXPECT_FLOAT_EQ(cfg.track_high_thresh, 0.25f);

    // Bare spec via resolve_tracker_config.
    TrackConfig resolved;
    EXPECT_TRUE(qyolo::track::resolve_tracker_config("ocsort", resolved));
    EXPECT_EQ(resolved.tracker_type, "ocsort");
    EXPECT_FALSE(qyolo::track::resolve_tracker_config("bogus", resolved));
}

TEST(QyoloTrackerFactory, CreatesAllSixAndRejectsUnknown) {
    const char* kTypes[] = {"bytetrack",  "botsort",   "ocsort",
                            "deepocsort", "fasttrack", "tracktrack"};
    for (const char* type : kTypes) {
        TrackConfig cfg;
        ASSERT_TRUE(default_tracker_config(type, cfg));
        auto tracker = create_tracker(cfg);
        ASSERT_NE(tracker, nullptr) << type;
    }
    TrackConfig cfg;
    default_tracker_config("bytetrack", cfg);
    cfg.tracker_type = "bogus";
    EXPECT_EQ(create_tracker(cfg), nullptr);
}

TEST(QyoloTrackerFactory, AcceptsReidOnFeatureFamily) {
    // Official with_reid semantics: the BoT-SORT family consumes the
    // FrameInput feature rows (model="auto" path); a featureless stream
    // degrades to motion-only association, so creation must succeed.
    TrackConfig cfg;
    default_tracker_config("botsort", cfg);
    cfg.with_reid = true;
    EXPECT_NE(create_tracker(cfg), nullptr);
    cfg = TrackConfig();
    default_tracker_config("tracktrack", cfg);
    cfg.with_reid = true;
    EXPECT_NE(create_tracker(cfg), nullptr);
}

#ifndef QYOLO_WITH_OPENCV
TEST(QyoloTrackerFactory, RejectsOpenCVOnlyGmcWithoutOpenCV) {
    TrackConfig cfg;
    default_tracker_config("botsort", cfg);
    cfg.gmc_method = "orb";
    EXPECT_EQ(create_tracker(cfg), nullptr);
}
#endif

TEST(QyoloTrackerReid, FeaturelessStreamDegradesGracefully) {
    // with_reid=true over a stream without features must run motion-only
    // association (upstream "feats missing" behavior): ids stay stable.
    std::vector<std::uint8_t> storage;
    const RgbFrame frame = grayFrame(64, 64, &storage);
    for (const char* type : {"botsort", "deepocsort", "tracktrack"}) {
        SCOPED_TRACE(type);
        TrackConfig cfg;
        ASSERT_TRUE(default_tracker_config(type, cfg));
        cfg.with_reid = true;
        auto tracker = create_tracker(cfg);
        ASSERT_NE(tracker, nullptr);
        int first_id = 0;
        for (int f = 0; f < 5; ++f) {
            FrameInput in;
            in.dets = {boxAt(10.0f + f * 2.0f, 10.0f, 20.0f + f * 2.0f, 20.0f,
                             0.9f, 0, 0)};
            in.frame = &frame;
            const std::vector<TrackedBox> tracks = tracker->update(in);
            ASSERT_EQ(tracks.size(), 1u);
            ASSERT_GT(tracks[0].track_id, 0);
            if (first_id == 0) {
                first_id = tracks[0].track_id;
            } else {
                EXPECT_EQ(tracks[0].track_id, first_id);
            }
        }
    }
}

TEST(QyoloTrackerReid, FeatureRowsKeepIdsStable) {
    // with_reid=true with per-detection feature rows: the association must
    // stay stable and id-consistent (the EMA/cosine path exercises
    // without crashing or producing duplicate ids).
    std::vector<std::uint8_t> storage;
    const RgbFrame frame = grayFrame(64, 64, &storage);
    TrackConfig cfg;
    ASSERT_TRUE(default_tracker_config("botsort", cfg));
    cfg.with_reid = true;
    auto tracker = create_tracker(cfg);
    ASSERT_NE(tracker, nullptr);
    int first_id = 0;
    for (int f = 0; f < 5; ++f) {
        FrameInput in;
        in.dets = {boxAt(10.0f + f * 2.0f, 10.0f, 20.0f + f * 2.0f, 20.0f, 0.9f,
                         0, 0)};
        in.frame = &frame;
        // Fixed appearance signature for the single object.
        in.feats = {{0.5f, 0.5f, 0.5f, 0.5f}};
        const std::vector<TrackedBox> tracks = tracker->update(in);
        ASSERT_EQ(tracks.size(), 1u);
        ASSERT_GT(tracks[0].track_id, 0);
        if (first_id == 0) {
            first_id = tracks[0].track_id;
        } else {
            EXPECT_EQ(tracks[0].track_id, first_id);
        }
    }
}

TEST(QyoloTrackerCore, StableIdAcrossFrames) {
    // One object moving right across frames keeps the same non-zero id.
    std::vector<std::uint8_t> storage;
    const RgbFrame frame = grayFrame(64, 64, &storage);
    for (const char* type : {"bytetrack", "botsort", "ocsort", "deepocsort",
                             "fasttrack", "tracktrack"}) {
        SCOPED_TRACE(type);
        TrackConfig cfg;
        ASSERT_TRUE(default_tracker_config(type, cfg));
        auto tracker = create_tracker(cfg);
        ASSERT_NE(tracker, nullptr);
        int first_id = 0;
        for (int f = 0; f < 5; ++f) {
            FrameInput in;
            in.dets = {boxAt(10.0f + f * 2.0f, 10.0f, 20.0f + f * 2.0f, 20.0f,
                             0.9f, 0, 0)};
            in.frame = &frame;
            const std::vector<TrackedBox> tracks = tracker->update(in);
            ASSERT_EQ(tracks.size(), 1u);
            ASSERT_GT(tracks[0].track_id, 0);
            if (first_id == 0) {
                first_id = tracks[0].track_id;
            } else {
                EXPECT_EQ(tracks[0].track_id, first_id);
            }
        }
    }
}

TEST(QyoloTrackerCore, DeterministicReplay) {
    // Feeding the same recorded detection sequence to a fresh tracker
    // twice produces identical output (the parity harness's replay
    // contract, ported to a unit test).
    const char* kTypes[] = {"bytetrack",  "botsort",   "ocsort",
                            "deepocsort", "fasttrack", "tracktrack"};
    for (const char* type : kTypes) {
        SCOPED_TRACE(type);
        std::vector<std::vector<TrackDet>> frames;
        for (int f = 0; f < 6; ++f) {
            const float drift = f * 1.5f;
            frames.push_back(
                    {boxAt(5 + drift, 5, 15 + drift, 15, 0.8f, 1, 0),
                     boxAt(30 - drift, 30, 40 - drift, 42, 0.6f, 0, 1),
                     boxAt(50, 8 + drift, 60, 18 + drift, 0.4f, 2, 2)});
        }
        auto run = [&]() {
            TrackConfig cfg;
            if (!default_tracker_config(type, cfg)) return std::string();
            auto tracker = create_tracker(cfg);
            if (!tracker) return std::string();
            // The id allocator is process-global by design (same as the
            // upstream C++ runtime), so replay determinism is asserted on
            // the ID ASSIGNMENT ORDER (first appearance -> 1, 2, ...)
            // plus the joined row and coordinates, never on absolute ids.
            std::string sig;
            char buf[64];
            std::vector<std::uint8_t> storage;
            const RgbFrame frame = grayFrame(64, 64, &storage);
            std::vector<int> id_map;  // raw id -> ordinal
            for (const auto& dets : frames) {
                FrameInput in;
                in.dets = dets;
                in.frame = &frame;
                for (const TrackedBox& t : tracker->update(in)) {
                    int ordinal = 0;
                    for (size_t i = 0; i < id_map.size(); ++i) {
                        if (id_map[i] == t.track_id)
                            ordinal = static_cast<int>(i) + 1;
                    }
                    if (ordinal == 0) {
                        id_map.push_back(t.track_id);
                        ordinal = static_cast<int>(id_map.size());
                    }
                    std::snprintf(buf, sizeof(buf), "%d:%d:%.4f;", ordinal,
                                  t.det.idx, static_cast<double>(t.det.cx));
                    sig += buf;
                }
                sig += "|";
            }
            return sig;
        };
        const std::string first = run();
        const std::string second = run();
        EXPECT_FALSE(first.empty());
        EXPECT_EQ(first, second);
    }
}

TEST(QyoloTrackerCore, ResetClearsState) {
    std::vector<std::uint8_t> storage;
    const RgbFrame frame = grayFrame(64, 64, &storage);
    TrackConfig cfg;
    ASSERT_TRUE(default_tracker_config("bytetrack", cfg));
    auto tracker = create_tracker(cfg);
    ASSERT_NE(tracker, nullptr);

    FrameInput in;
    in.dets = {boxAt(10, 10, 20, 20)};
    in.frame = &frame;
    const auto first = tracker->update(in);
    ASSERT_EQ(first.size(), 1u);
    ASSERT_GT(first[0].track_id, 0);

    // Reset + a new object at a different position must start a fresh id
    // table (the id may be reused — what matters is a clean re-init that
    // does not remember the previous track's Kalman state).
    tracker->reset();
    FrameInput fresh;
    fresh.dets = {boxAt(40, 40, 50, 50)};
    fresh.frame = &frame;
    const auto second = tracker->update(fresh);
    ASSERT_EQ(second.size(), 1u);
    EXPECT_GT(second[0].track_id, 0);
}

TEST(QyoloTrackerCore, ObbAnglePassThrough) {
    // Angled detections keep their angle in the tracked output (obb rows
    // stay rotated through the tracker).
    std::vector<std::uint8_t> storage;
    const RgbFrame frame = grayFrame(64, 64, &storage);
    TrackConfig cfg;
    ASSERT_TRUE(default_tracker_config("bytetrack", cfg));
    auto tracker = create_tracker(cfg);
    ASSERT_NE(tracker, nullptr);

    TrackDet t = boxAt(10, 10, 30, 20);
    t.angle = 0.5f;  // rotated row
    FrameInput in;
    in.dets = {t};
    in.frame = &frame;
    const auto tracks = tracker->update(in);
    ASSERT_EQ(tracks.size(), 1u);
    EXPECT_TRUE(tracks[0].det.angled());
    EXPECT_NEAR(tracks[0].det.angle, 0.5f, 1e-6f);
}

TEST(QyoloTrackerCore, TrackTrackPromotesOnSecondUpdate) {
    // Official TTSTrack confirmation rule (track_tracker.py): a track with
    // min_track_len=3 becomes Tracked/is_activated once its history reaches
    // min_track_len entries — i.e. on the SECOND update after activation
    // (tracklet_len + 1 >= min_track_len), not the third.
    std::vector<std::uint8_t> storage;
    const RgbFrame frame = grayFrame(64, 64, &storage);
    TrackConfig cfg;
    ASSERT_TRUE(default_tracker_config("tracktrack", cfg));
    ASSERT_EQ(cfg.min_track_len, 3);
    auto tracker = create_tracker(cfg);
    ASSERT_NE(tracker, nullptr);

    // Frame 1: nothing (the track must be born at frame_id != 1 so it does
    // not get the frame-1 is_activated head start).
    FrameInput empty;
    empty.frame = &frame;
    EXPECT_EQ(tracker->update(empty).size(), 0u);

    // Frame 2: activation frame — state New, is_activated false.
    FrameInput in;
    in.dets = {boxAt(10, 10, 20, 20)};
    in.frame = &frame;
    EXPECT_EQ(tracker->update(in).size(), 0u);

    // Frame 3: first update — still one entry short of min_track_len.
    in.dets = {boxAt(11, 10, 21, 20)};
    EXPECT_EQ(tracker->update(in).size(), 0u);

    // Frame 4: second update — tracklet_len + 1 reaches min_track_len and
    // the track must be emitted now.
    in.dets = {boxAt(12, 10, 22, 20)};
    const auto tracks = tracker->update(in);
    ASSERT_EQ(tracks.size(), 1u);
    EXPECT_GT(tracks[0].track_id, 0);
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
