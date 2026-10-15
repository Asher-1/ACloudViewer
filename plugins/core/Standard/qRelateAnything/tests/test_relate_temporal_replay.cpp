// SPDX-License-Identifier: MIT
// Replays a recorded detection / raw-logit scenario through the C++ port of
// upstream's tracker and relation-belief filter and prints per-frame state in
// the same text format as tests/compare_temporal_upstream.py, which diffs it
// against upstream deploy/render_video.py. Qt-free.
//
//   test_relate_temporal_replay scenario.txt > cpp.txt
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "RelateAnythingTemporal.h"

using namespace RelateTemporal;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s scenario.txt\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::vector<std::string> names;
    std::vector<std::uint8_t> spatial;
    float calA = 1, calB = 0;
    int W = 0, H = 0, frames = 0;
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
            ls >> v;
            for (int i = 0; i < v; ++i) {
                next(line);
                const auto bar = line.find('|');
                names.push_back(line.substr(0, bar));
                spatial.push_back(line.substr(bar + 1) == "1");
            }
            book = std::make_unique<EdgeBook>(params, names, spatial);
        } else if (tag == "CAL") {
            ls >> calA >> calB;
        } else if (tag == "SIZE") {
            ls >> W >> H;
        } else if (tag == "FRAMES") {
            ls >> frames;
        } else if (tag == "FRAME") {
            int index, n;
            ls >> index >> n;
            std::vector<Box> boxes;
            std::vector<std::string> labels;
            std::vector<float> confs;
            for (int i = 0; i < n; ++i) {
                next(line);
                std::istringstream ds(line);
                std::string t, label;
                Box b;
                float c;
                ds >> t >> b[0] >> b[1] >> b[2] >> b[3] >> c >> label;
                boxes.push_back(b);
                labels.push_back(label);
                confs.push_back(c);
            }
            next(line);
            std::istringstream rs(line);
            std::string rt;
            std::size_t K;
            rs >> rt >> K;
            const std::size_t V = names.size();
            std::vector<float> pred(K * V), pair(K);
            std::vector<std::int32_t> sub(K), obj(K), valid(K);
            for (std::size_t k = 0; k < K; ++k) {
                next(line);
                std::istringstream ps(line);
                std::string pt;
                ps >> pt >> sub[k] >> obj[k] >> valid[k] >> pair[k];
                for (std::size_t v = 0; v < V; ++v) ps >> pred[k * V + v];
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
    return 0;
}
