// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "RelateAnythingTemporal.h"

using namespace RelateTemporal;

#define REQUIRE(c)                                                         \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::fprintf(stderr, "FAILED %s:%d  %s\n", __FILE__, __LINE__, \
                         #c);                                              \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

namespace {
const Box kA{10, 10, 110, 110}, kB{200, 200, 300, 300};

struct Frame {
    float pred[2], pair[1];
    std::int32_t sub[1] = {0}, obj[1] = {1}, valid[1] = {1};
    RawFrame raw() const {
        RawFrame r;
        r.predLogits = pred;
        r.pairLogits = pair;
        r.sub = sub;
        r.obj = obj;
        r.valid = valid;
        r.K = 1;
        r.V = 2;
        return r;
    }
};
}  // namespace

int main() {
    Params p;
    // Identity persists while the box moves; a label flip is out-voted.
    {
        Tracker t(p);
        int tid = -1;
        for (int f = 0; f < 6; ++f) {
            Box a = kA;
            a[0] += f * 4;
            a[2] += f * 4;
            const auto m = t.observe(
                    {a, kB}, {f == 3 ? "man" : "person", "horse"}, {.9f, .9f});
            REQUIRE(m.size() == 2);
            if (tid < 0) tid = m.at(0);
            REQUIRE(m.at(0) == tid);
            t.step(640, 480);
        }
        REQUIRE(t.tracks.size() == 2);
        REQUIRE(t.byId(tid)->label() == "person");
        REQUIRE(t.byId(tid)->alpha > 0);  // visible after minHits
        // A track survives maxAge misses and is dropped afterwards.
        for (int f = 0; f < p.maxAge; ++f) t.observe({}, {}, {});
        REQUIRE(t.tracks.size() == 2);
        t.observe({}, {}, {});
        REQUIRE(t.tracks.empty());
    }
    // Duplicate regions (same pixels, two names) collapse to one track.
    {
        Tracker t(p);
        const auto m = t.observe({kA, {11, 11, 111, 111}}, {"man", "person"},
                                 {.9f, .8f});
        REQUIRE(m.size() == 1 && t.tracks.size() == 1);
    }
    // A low-confidence detection keeps an existing identity (second stage) but
    // never starts a track.
    {
        Tracker t(p);
        const auto first = t.observe({kA}, {"x"}, {.9f});
        const auto second = t.observe({kA}, {"x"}, {.1f});
        REQUIRE(second.at(0) == first.at(0));
        Tracker u(p);
        REQUIRE(u.observe({kA}, {"x"}, {.1f}).empty());
    }
    // Belief + hysteresis: an edge turns on after onFrames, off after
    // offFrames.
    {
        Tracker t(p);
        EdgeBook book(p, {"riding", "beside"}, {0, 1});
        Frame hi;
        hi.pred[0] = 3;
        hi.pred[1] = -3;
        hi.pair[0] = 2;
        std::map<int, int> d2t;
        for (int f = 0; f < 3; ++f) {
            d2t = t.observe({kA, kB}, {"person", "horse"}, {.9f, .9f});
            t.step(640, 480);
            book.observe(d2t, {.9f, .9f}, hi.raw());
            std::set<int> live;
            for (const auto& tr : t.tracks) live.insert(tr.tid);
            book.step(live);
        }
        auto vis = book.visible();
        REQUIRE(vis.size() == 1 && book.name(vis[0].pred) == "riding");
        REQUIRE(!vis[0].spatial && vis[0].score > .9f);
        // Unmeasurable (object not detected): belief widens, score is held.
        const double before = book.edges[0].score;
        const double pBefore = book.edges[0].bel.P;
        std::map<int, int> onlySubject = {{0, d2t.at(0)}};
        book.observe(onlySubject, {.9f, .9f}, hi.raw());
        REQUIRE(std::fabs(book.edges[0].score - before) < 1e-6);
        REQUIRE(book.edges[0].bel.P > pBefore);
        // Strong negative evidence turns it off after offFrames observations.
        Frame lo;
        lo.pred[0] = -6;
        lo.pred[1] = -6;
        lo.pair[0] = -2;
        for (int f = 0; f < p.offFrames + 2; ++f) {
            d2t = t.observe({kA, kB}, {"person", "horse"}, {.9f, .9f});
            t.step(640, 480);
            book.observe(d2t, {.9f, .9f}, lo.raw());
        }
        bool anyOn = false;
        for (const auto& e : book.edges) anyOn |= e.on;
        REQUIRE(!anyOn);
        std::set<int> live;
        for (const auto& tr : t.tracks) live.insert(tr.tid);
        for (int f = 0; f < 12; ++f) book.step(live);
        REQUIRE(book.visible().empty());
    }
    // Spatial edges are capped and never displace interactions.
    {
        Params q = p;
        q.maxEdges = 3;
        q.maxSpatial = 1;
        EdgeBook book(q, {"a", "b", "c", "on", "near"}, {0, 0, 0, 1, 1});
        for (int i = 0; i < 5; ++i) {
            Edge e{0, i + 1, i, .9f - .01f * i, i >= 3};
            e.alpha = 1;
            e.shown = e.score;
            book.edges.push_back(e);
        }
        const auto v = book.visible();
        int spatial = 0, inter = 0;
        for (const auto& e : v) (e.spatial ? spatial : inter)++;
        REQUIRE(spatial == 1 && inter == 2);
    }
    std::puts("relate temporal tests passed");
    return 0;
}
