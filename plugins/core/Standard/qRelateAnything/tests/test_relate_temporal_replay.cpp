// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "RelateAnythingTemporal.h"

using namespace RelateTemporal;

namespace {
std::string scenarioPath;
bool invalidScenarioInput = false;

// stdout is the stable numeric replay protocol consumed by
// compare_temporal_upstream.py. GoogleTest diagnostics must not alter it.
class StderrResultPrinter : public ::testing::EmptyTestEventListener {
    void OnTestPartResult(const ::testing::TestPartResult& result) override {
        if (result.failed())
            std::fprintf(stderr, "%s:%d: %s\n",
                         result.file_name() ? result.file_name() : "replay",
                         result.line_number(), result.summary());
    }
    void OnTestEnd(const ::testing::TestInfo& info) override {
        std::fprintf(stderr, "%s %s.%s\n",
                     info.result()->Passed() ? "PASS" : "FAIL",
                     info.test_suite_name(), info.name());
    }
};

TEST(RelateTemporalReplay, ReplaysCompleteScenarioInUpstreamOperationOrder) {
    invalidScenarioInput = scenarioPath.empty();
    ASSERT_FALSE(scenarioPath.empty())
            << "Pass scenario.txt after GoogleTest options";
    std::ifstream in(scenarioPath);
    invalidScenarioInput = !in.is_open();
    ASSERT_TRUE(in.is_open()) << "Cannot open " << scenarioPath;
    std::vector<std::string> names;
    std::vector<std::uint8_t> spatial;
    float calA = 1, calB = 0;
    int W = 0, H = 0, frames = 0, replayedFrames = 0;
    std::string line;
    const auto next = [&](std::string& l) {
        while (std::getline(in, l))
            if (!l.empty()) return true;
        return false;
    };
    Params params;  // defaults == upstream render_video.py defaults
    std::unique_ptr<EdgeBook> book;
    Tracker tracker(params);
    while (next(line)) {
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;
        if (tag == "PRED") {
            int v;
            ASSERT_TRUE(bool(ls >> v));
            ASSERT_GT(v, 0);
            for (int i = 0; i < v; ++i) {
                ASSERT_TRUE(next(line));
                const auto bar = line.find('|');
                ASSERT_NE(bar, std::string::npos);
                names.push_back(line.substr(0, bar));
                spatial.push_back(line.substr(bar + 1) == "1");
            }
            book = std::make_unique<EdgeBook>(params, names, spatial);
        } else if (tag == "CAL") {
            ASSERT_TRUE(bool(ls >> calA >> calB));
        } else if (tag == "SIZE") {
            ASSERT_TRUE(bool(ls >> W >> H));
        } else if (tag == "FRAMES") {
            ASSERT_TRUE(bool(ls >> frames));
            ASSERT_GE(frames, 0);
        } else if (tag == "FRAME") {
            int index, n;
            ASSERT_TRUE(bool(ls >> index >> n));
            ASSERT_GE(index, 0);
            ASSERT_GE(n, 0);
            ASSERT_GT(W, 0);
            ASSERT_GT(H, 0);
            ASSERT_NE(book, nullptr) << "PRED must precede FRAME";
            ++replayedFrames;
            std::vector<Box> boxes;
            std::vector<std::string> labels;
            std::vector<float> confs;
            for (int i = 0; i < n; ++i) {
                ASSERT_TRUE(next(line));
                std::istringstream ds(line);
                std::string t, label;
                Box b;
                float c;
                ASSERT_TRUE(bool(ds >> t >> b[0] >> b[1] >> b[2] >> b[3] >> c >>
                                 label));
                ASSERT_EQ(t, "D");
                boxes.push_back(b);
                labels.push_back(label);
                confs.push_back(c);
            }
            ASSERT_TRUE(next(line));
            std::istringstream rs(line);
            std::string rt;
            std::size_t K;
            ASSERT_TRUE(bool(rs >> rt >> K));
            ASSERT_EQ(rt, "R");
            const std::size_t V = names.size();
            std::vector<float> pred(K * V), pair(K);
            std::vector<std::int32_t> sub(K), obj(K), valid(K);
            for (std::size_t k = 0; k < K; ++k) {
                ASSERT_TRUE(next(line));
                std::istringstream ps(line);
                std::string pt;
                ASSERT_TRUE(bool(ps >> pt >> sub[k] >> obj[k] >> valid[k] >>
                                 pair[k]));
                ASSERT_EQ(pt, "P");
                for (std::size_t v = 0; v < V; ++v)
                    ASSERT_TRUE(bool(ps >> pred[k * V + v]));
            }
            // Same order as upstream render_one with --stride 1:
            // tracker.observe, book.observe, tracker.step, book.step.
            const auto det2track =
                    tracker.observe(boxes, labels, confs, double(W) * H, .85f);
            RawFrame raw;
            if (n >= 2) {  // relate() returns (no triplets, no raw) below 2
                raw.predLogits = pred.data();
                raw.pairLogits = pair.data();
                raw.sub = sub.data();
                raw.obj = obj.data();
                raw.valid = valid.data();
                raw.K = K;
                raw.V = V;
                raw.calA = calA;
                raw.calB = calB;
            }
            book->observe(det2track, confs, raw);
            tracker.step(W, H);
            std::set<int> live;
            for (const auto& t : tracker.tracks) live.insert(t.tid);
            book->step(live);

            std::printf("FRAME %d\n", index);
            std::vector<const Track*> ts;
            for (const auto& t : tracker.tracks) ts.push_back(&t);
            std::sort(ts.begin(), ts.end(), [](const Track* a, const Track* b) {
                return a->tid < b->tid;
            });
            for (const auto* t : ts) {
                const Box b = t->box();
                std::printf("T %d %.3f %.3f %.3f %.3f %.3f %d %d %s\n", t->tid,
                            b[0], b[1], b[2], b[3], t->alpha, t->hits,
                            t->misses, t->label().c_str());
            }
            std::vector<Edge> es = book->edges;
            std::sort(es.begin(), es.end(), [](const Edge& a, const Edge& b) {
                return std::tie(a.sub, a.obj, a.pred) <
                       std::tie(b.sub, b.obj, b.pred);
            });
            for (const auto& e : es)
                std::printf("E %d %d %d %.3f %.3f %d %.3f %d %d\n", e.sub,
                            e.obj, e.pred, e.score, e.shown, e.on ? 1 : 0,
                            e.alpha, e.onRun, e.offRun);
            for (const auto& e : book->visible())
                std::printf("V %d %d %d\n", e.sub, e.obj, e.pred);
        }
    }
    EXPECT_EQ(replayedFrames, frames)
            << "Every declared frame must be replayed";
}
}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (argc == 2) scenarioPath = argv[1];
    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new StderrResultPrinter);
    const int result = RUN_ALL_TESTS();
    return result != 0 && invalidScenarioInput ? 2 : result;
}
