// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// Qt-light decision logic shared by every RelateAnything front end (the
// standalone qRelateAnything worker and the qAnyLabeling workspace), so both
// pick objects and edges identically. It mirrors the official
// RelateAnything-GGML deploy/render_video.py and demo_webcam.py: class-agnostic
// dedupe, near-full-frame removal, score calibration, per-stream floors,
// one edge per (object, predicate) and per (subject, object, leading word).
#include <QSize>
#include <QString>
#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>
#include <vector>

#include "aicore/relation_capi.h"

namespace RelateDecode {

struct Candidate {
    float box[4];
    float score;
    QString name;
    int source;
};

inline float boxIou(const float* a, const float* b) {
    const float iw = std::min(a[2], b[2]) - std::max(a[0], b[0]);
    const float ih = std::min(a[3], b[3]) - std::max(a[1], b[1]);
    if (iw <= 0 || ih <= 0) return 0;
    const float inter = iw * ih;
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

// Best maxBoxes objects after upstream's class-agnostic NMS at IoU 0.7: a big
// vocabulary returns one object under several names (horse / pony / animal).
// Image path: dedupe at IoU 0.7 (make_reel.py) and drop near-full-frame boxes.
// Video path: upstream render_video.py feeds the relation head every detection
// (only capped by --max_boxes) and lets the tracker apply --merge_iou and
// --max_box_frac, so both are switched off there (dedupeIou > 1, fraction 1).
inline std::vector<Candidate> keepBest(std::vector<Candidate> found,
                                       int maxBoxes,
                                       const QSize& image,
                                       float maxFraction,
                                       float dedupeIou) {
    const double area = double(image.width()) * image.height();
    if (area > 0 && maxFraction < 1.0f)
        found.erase(std::remove_if(found.begin(), found.end(),
                                   [&](const Candidate& c) {
                                       return (c.box[2] - c.box[0]) *
                                                      (c.box[3] - c.box[1]) /
                                                      area >
                                              maxFraction;
                                   }),
                    found.end());
    std::stable_sort(found.begin(), found.end(),
                     [](const Candidate& a, const Candidate& b) {
                         return a.score > b.score;
                     });
    std::vector<Candidate> kept;
    for (const auto& c : found) {
        bool duplicate = false;
        for (const auto& k : kept)
            if (boxIou(c.box, k.box) > dedupeIou) {
                duplicate = true;
                break;
            }
        if (!duplicate) kept.push_back(c);
        if (int(kept.size()) >= maxBoxes) break;
    }
    return kept;
}

// The decomposed streams carry sigmoid(logit) without the GGUF calibration
// (upstream's native path does the same); the flat list and upstream's Python
// contract use sigmoid(a * logit + b). The transform is monotone, so ranking
// is unchanged; scores are clamped so float saturation near 1 stays finite.
inline float calibrated(float score, float a, float b) {
    const float p = std::clamp(score, 1e-6f, 1.0f - 1e-6f);
    const float z = a * std::log(p / (1.0f - p)) + b;
    return 1.0f / (1.0f + std::exp(-std::clamp(z, -80.0f, 80.0f)));
}

struct PickedEdge {
    aicore_relation_edge edge;
    bool spatial;
    float rank = 0;  // score * conf(sub) * conf(obj) when weighting is on
};

struct Params {
    float relationThreshold = .45f;  // interaction floor
    float spatialThreshold = .42f;   // spatial floor
    bool boxScoreWeight = true;      // rank by score * conf(sub) * conf(obj)
    int birthTopk = 24;              // hypothesis pool of the decomposed path
    int maxRelations = 8;
    int maxSpatial = 3;
};

// Mirrors upstream deploy/render_video.py: interaction edges need
// relationThreshold, spatial edges spatialThreshold; one edge per
// (object, predicate) and per (subject, object, leading word); spatial edges
// are capped and interactions get first call on the remaining budget.
inline std::vector<PickedEdge> pickEdges(const aicore_relation_result& r,
                                         const Params& s,
                                         bool decomposed,
                                         float calibrationA,
                                         float calibrationB,
                                         const std::vector<float>& confs) {
    std::vector<PickedEdge> all;
    const auto conf = [&](uint32_t i) {
        return i < confs.size() ? confs[i] : 1.f;
    };
    if (decomposed) {
        for (size_t i = 0; i < r.semantic_count; ++i) {
            auto e = r.semantic_relations[i];
            e.score = calibrated(e.score, calibrationA, calibrationB);
            if (e.score >= s.relationThreshold) all.push_back({e, false});
        }
        for (size_t i = 0; i < r.spatial_count; ++i) {
            auto e = r.spatial_relations[i];
            e.score = calibrated(e.score, calibrationA, calibrationB);
            if (e.score >= s.spatialThreshold) all.push_back({e, true});
        }
    } else {
        for (size_t i = 0; i < r.relation_count; ++i)
            if (r.relations[i].score >= s.relationThreshold)
                all.push_back({r.relations[i], false});
    }
    // Upstream decode(): the floor applies to the raw relation score, but the
    // top-k is taken by score * conf(sub) * conf(obj) (box_score_weight), so a
    // pair built on shaky boxes cannot crowd out a solid one. Video births use
    // --topk (24) hypotheses; the flat image demo keeps maxRelations.
    for (auto& p : all)
        p.rank = p.edge.score *
                 (s.boxScoreWeight ? conf(p.edge.subject) * conf(p.edge.object)
                                   : 1.f);
    std::stable_sort(all.begin(), all.end(),
                     [](const PickedEdge& a, const PickedEdge& b) {
                         return a.rank > b.rank;
                     });
    const size_t pool =
            size_t(std::max(1, decomposed ? s.birthTopk : s.maxRelations));
    if (all.size() > pool) all.resize(pool);
    // Display order (upstream EdgeBook.visible) is by raw strength.
    std::stable_sort(all.begin(), all.end(),
                     [](const PickedEdge& a, const PickedEdge& b) {
                         return a.edge.score > b.edge.score;
                     });
    std::set<std::pair<uint32_t, uint32_t>> byObject, byPair;
    std::set<std::tuple<uint32_t, uint32_t, QString>> byHead;
    std::vector<PickedEdge> interaction, spatial;
    for (const auto& p : all) {
        const auto& e = p.edge;
        const QString name = QString::fromUtf8(r.predicate_names[e.predicate]);
        if (!decomposed) {
            if (!byPair.insert({e.subject, e.object}).second) continue;
        } else {
            if (!byObject.insert({e.object, e.predicate}).second) continue;
            if (!byHead.insert({e.subject, e.object, name.section(' ', 0, 0)})
                         .second)
                continue;
        }
        (p.spatial ? spatial : interaction).push_back(p);
    }
    if (int(spatial.size()) > s.maxSpatial) spatial.resize(s.maxSpatial);
    const int room = std::max(0, s.maxRelations - int(spatial.size()));
    if (int(interaction.size()) > room) interaction.resize(room);
    std::vector<PickedEdge> picked = interaction;
    picked.insert(picked.end(), spatial.begin(), spatial.end());
    std::stable_sort(picked.begin(), picked.end(),
                     [](const PickedEdge& a, const PickedEdge& b) {
                         return a.edge.score > b.edge.score;
                     });
    if (int(picked.size()) > s.maxRelations) picked.resize(s.maxRelations);
    return picked;
}

}  // namespace RelateDecode
