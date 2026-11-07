// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Track-parity harness: consumes the SAME per-frame detection rows the
// official Python runtime was fed (stdin, one line per frame) and emits the
// tracked rows so the checker can diff field-by-field against the official
// tracker output. Not a ctest — driven by tests/track_parity_check.py.
//
// stdin frame line: <frame> <n>  then n groups of <cx cy w h score cls idx>
// stdin tracker line before the first frame: TRACK <type>
// stdout frame line: F<frame> [x1,y1,x2,y2,id,score,cls,idx] ...

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "tracking/tracker.hpp"

using namespace qyolo::track;

int main() {
    std::string cmd;
    Tracker* tracker = nullptr;
    TrackConfig cfg;
    while (std::cin.peek() != EOF) {
        std::string tag;
        if (!(std::cin >> tag)) break;
        if (tag == "TRACK") {
            std::string type;
            std::cin >> type;
            if (tracker != nullptr) delete tracker;
            cfg = TrackConfig();
            if (!default_tracker_config(type, cfg)) {
                std::fprintf(stderr, "unknown tracker type %s\n", type.c_str());
                return 2;
            }
            tracker = create_tracker(cfg).release();
            if (tracker == nullptr) {
                std::fprintf(stderr, "tracker creation failed for %s\n",
                             type.c_str());
                return 2;
            }
            continue;
        }
        if (tag != "F") {
            std::fprintf(stderr, "unexpected token %s\n", tag.c_str());
            return 2;
        }
        int frame = 0, n = 0;
        std::cin >> frame >> n;
        FrameInput in;
        for (int k = 0; k < n; ++k) {
            TrackDet d;  // axis-aligned; no GMC frames in this harness
            d.angle = -10.0f;
            std::cin >> d.cx >> d.cy >> d.w >> d.h >> d.score >> d.class_id >>
                    d.idx;
            in.dets.push_back(d);
        }
        std::printf("F%d", frame);
        for (const TrackedBox& t : tracker->update(in)) {
            const TrackDet& d = t.det;
            const float x1 = d.cx - d.w / 2, y1 = d.cy - d.h / 2;
            std::printf(" [%.4f,%.4f,%.4f,%.4f,%d,%.4f,%d,%d]", x1, y1, d.w,
                        d.h, t.track_id, d.score, d.class_id, d.idx);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    delete tracker;
    return 0;
}
