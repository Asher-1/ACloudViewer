// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// Temporal stabilisation for video, ported from upstream
// deploy/render_video.py (github.com/Maelic/RelateAnything, AGPL-3.0): the
// relation head is per-frame and has no memory, so boxes jitter, labels flip
// and relations blink. Four stages, all outside the model:
//   Tracker       two-stage IoU association over Kalman-filtered boxes
//   RelationBelief 1-D Kalman filter on the calibrated log-odds of a relation
//   EdgeBook      belief (or EMA) + hysteresis + alpha fades per relation
//   visible()     one edge per (object, predicate) / (pair, leading word),
//                 interactions first, spatial edges capped
// Qt-free on purpose so it can be unit tested without a display.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace RelateTemporal {

// Every field mirrors an upstream flag or class default (names in comments).
struct Params {
    // Tracker
    float trackHigh = .25f;   // --track_high
    float iouThr = .3f;       // --iou_thr
    float lowIouThr = .5f;    // Tracker second-stage gate
    float mergeIou = .75f;    // --merge_iou
    int maxAge = 8;           // --max_age
    int minHits = 2;          // --min_hits
    float kfQ = 1.f;          // --kf_q  (box process noise)
    float kfR = 10.f;         // --kf_r  (box measurement noise)
    float trackFade = .18f;   // Tracker.step(fade=0.18)
    // EdgeBook
    bool belief = true;       // not --no_belief
    float scoreEma = .35f;    // --score_ema (only without belief)
    float onThr = .45f;       // --on_thr
    float spatialOnThr = .42f;  // --spatial_on_thr
    float offThr = .38f;      // --off_thr (clamped to onThr - 0.10)
    int onFrames = 2;         // --on_frames
    int offFrames = 6;        // --off_frames
    float edgeFade = .12f;    // EdgeBook(fade=0.12)
    float relQ = .05f;        // --kf_rel_q
    float relR = 1.f;         // --kf_rel_r
    float pMax = 8.f;         // EdgeBook(p_max=8.0)
    int maxEdges = 8;         // --max_edges
    int maxSpatial = 3;       // --max_spatial
    int maxPerPair = 2;       // --max_per_pair (1 = best predicate per pair)
    int birthTopk = 24;       // --topk (hypotheses promoted per frame)

    float effectiveOffThr() const { return std::min(offThr, onThr - .10f); }
    bool same(const Params& o) const {
        return std::tie(trackHigh, iouThr, lowIouThr, mergeIou, maxAge, minHits,
                        kfQ, kfR, trackFade, belief, scoreEma, onThr,
                        spatialOnThr, offThr, onFrames, offFrames, edgeFade,
                        relQ, relR, pMax, maxEdges, maxSpatial, maxPerPair,
                        birthTopk) ==
               std::tie(o.trackHigh, o.iouThr, o.lowIouThr, o.mergeIou,
                        o.maxAge, o.minHits, o.kfQ, o.kfR, o.trackFade,
                        o.belief, o.scoreEma, o.onThr, o.spatialOnThr, o.offThr,
                        o.onFrames, o.offFrames, o.edgeFade, o.relQ, o.relR,
                        o.pMax, o.maxEdges, o.maxSpatial, o.maxPerPair,
                        o.birthTopk);
    }
};

using Box = std::array<float, 4>;  // x1 y1 x2 y2

inline float iou(const Box& a, const Box& b) {
    const float iw = std::min(a[2], b[2]) - std::max(a[0], b[0]);
    const float ih = std::min(a[3], b[3]) - std::max(a[1], b[1]);
    if (iw <= 0 || ih <= 0) return 0;
    const float inter = iw * ih;
    const float areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const float areaB = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / std::max(areaA + areaB - inter, 1e-6f);
}

// Constant-velocity Kalman filter on (cx, cy, w, h). The upstream matrices
// are block diagonal and the position/velocity cross terms start (and stay)
// at zero, so the velocity rows of the gain are zero: each component is an
// independent scalar filter whose velocity state never leaves 0. Reproduced
// as written; only the position variance sees the velocity variance growth.
class KalmanBox {
public:
    KalmanBox(const Box& b, float q, float r) : q_(q), r_(r) {
        x_[0] = (b[0] + b[2]) / 2;
        x_[1] = (b[1] + b[3]) / 2;
        x_[2] = b[2] - b[0];
        x_[3] = b[3] - b[1];
        for (int i = 0; i < 4; ++i) {
            v_[i] = 0;
            p_[i] = 10.;
            pv_[i] = 1e3;
        }
    }
    void predict() {
        for (int i = 0; i < 4; ++i) {
            x_[i] += v_[i];
            p_[i] += pv_[i] + q_;
            pv_[i] += q_ * .01;
        }
    }
    void update(const Box& b) {
        const double z[4] = {(b[0] + b[2]) / 2., (b[1] + b[3]) / 2.,
                             double(b[2] - b[0]), double(b[3] - b[1])};
        for (int i = 0; i < 4; ++i) {
            const double k = p_[i] / (p_[i] + r_);
            x_[i] += k * (z[i] - x_[i]);
            p_[i] -= k * p_[i];
        }
        x_[2] = std::max(x_[2], 2.);
        x_[3] = std::max(x_[3], 2.);
    }
    Box box() const {
        return {float(x_[0] - x_[2] / 2), float(x_[1] - x_[3] / 2),
                float(x_[0] + x_[2] / 2), float(x_[1] + x_[3] / 2)};
    }
    void setBox(const Box& b) {
        x_[0] = (b[0] + b[2]) / 2;
        x_[1] = (b[1] + b[3]) / 2;
        x_[2] = b[2] - b[0];
        x_[3] = b[3] - b[1];
    }

private:
    double x_[4], v_[4], p_[4], pv_[4];
    double q_, r_;
};

struct Track {
    int tid;
    KalmanBox kf;
    float conf;
    std::vector<std::pair<std::string, int>> votes;  // insertion ordered
    int hits = 0, misses = 0;
    float alpha = 0;
    Box box() const { return kf.box(); }
    void vote(const std::string& label) {
        for (auto& v : votes)
            if (v.first == label) {
                ++v.second;
                return;
            }
        votes.emplace_back(label, 1);
    }
    // Majority vote (first inserted wins ties), not the last frame's answer:
    // the open-vocabulary detector flips "man" / "person" on one region.
    std::string label() const {
        const std::pair<std::string, int>* best = nullptr;
        for (const auto& v : votes)
            if (!best || v.second > best->second) best = &v;
        return best ? best->first : std::string("?");
    }
};

class Tracker {
public:
    explicit Tracker(const Params& p) : p_(p) {}
    std::vector<Track> tracks;

    // detection index -> track id. Detections are never compacted: a relation
    // addresses its endpoints by index into the detector's own output.
    std::map<int, int> observe(const std::vector<Box>& boxes,
                               const std::vector<std::string>& labels,
                               const std::vector<float>& confs,
                               double frameArea = 0,
                               float maxFrac = 1.f) {
        const int n = int(boxes.size());
        std::vector<char> ok(size_t(n), 1);
        if (frameArea > 0 && maxFrac < 1.f)
            for (int i = 0; i < n; ++i)
                ok[size_t(i)] = (boxes[size_t(i)][2] - boxes[size_t(i)][0]) *
                                        (boxes[size_t(i)][3] -
                                         boxes[size_t(i)][1]) / frameArea <=
                                maxFrac;
        // Cross-class duplicate suppression: "man" and "person" over the same
        // pixels survive class-aware NMS and would draw every relation twice.
        std::vector<int> order;
        order.resize(size_t(n));
        for (int i = 0; i < n; ++i) order[size_t(i)] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return confs[size_t(a)] > confs[size_t(b)];
        });
        std::vector<int> keep, hi, lo;
        for (int i : order) {
            if (!ok[size_t(i)]) continue;
            bool dup = false;
            for (int k : keep)
                if (iou(boxes[size_t(i)], boxes[size_t(k)]) > p_.mergeIou) {
                    dup = true;
                    break;
                }
            if (!dup) keep.push_back(i);
        }
        for (int d : keep)
            (confs[size_t(d)] >= p_.trackHigh ? hi : lo).push_back(d);

        std::map<int, int> det2track;
        std::vector<Track*> all;
        for (auto& t : tracks) all.push_back(&t);
        std::vector<std::pair<int, Track*>> matched;
        std::vector<int> unHi;
        std::vector<Track*> unTracks;
        match(hi, boxes, all, p_.iouThr, matched, unHi, unTracks);
        // second stage: still-unmatched tracks against low-confidence leftovers
        std::vector<std::pair<int, Track*>> matched2;
        std::vector<int> ignoredLo;
        std::vector<Track*> stillUnmatched;
        match(lo, boxes, unTracks, p_.lowIouThr, matched2, ignoredLo,
              stillUnmatched);
        matched.insert(matched.end(), matched2.begin(), matched2.end());
        for (auto& m : matched) {
            const int d = m.first;
            Track* t = m.second;
            t->kf.update(boxes[size_t(d)]);
            t->conf = confs[size_t(d)];
            t->vote(labels[size_t(d)]);
            ++t->hits;
            t->misses = 0;
            det2track[d] = t->tid;
        }
        for (int d : unHi) {  // confident and unexplained: new track
            Track t{next_, KalmanBox(boxes[size_t(d)], p_.kfQ, p_.kfR),
                    confs[size_t(d)]};
            t.vote(labels[size_t(d)]);
            t.hits = 1;
            det2track[d] = t.tid;
            ++next_;
            tracks.push_back(std::move(t));
        }
        for (Track* t : stillUnmatched) ++t->misses;
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [&](const Track& t) {
                                        return t.misses > p_.maxAge;
                                    }),
                     tracks.end());
        return det2track;
    }

    // Once per RENDERED frame: the Kalman predicts on every one.
    void step(int W, int H) {
        for (auto& t : tracks) {
            t.kf.predict();
            Box b = t.kf.box();
            b[0] = std::min(std::max(b[0], 0.f), float(W - 2));
            b[1] = std::min(std::max(b[1], 0.f), float(H - 2));
            b[2] = std::min(std::max(b[2], b[0] + 2), float(W));
            b[3] = std::min(std::max(b[3], b[1] + 2), float(H));
            t.kf.setBox(b);
            const bool vis = t.hits >= p_.minHits && t.misses <= 2;
            t.alpha = vis ? std::min(1.f, t.alpha + p_.trackFade)
                          : std::max(0.f, t.alpha - p_.trackFade);
        }
    }
    const Track* byId(int tid) const {
        for (const auto& t : tracks)
            if (t.tid == tid) return &t;
        return nullptr;
    }

private:
    // Greedy IoU matching on descending IoU (ties: larger indices first, as
    // upstream's reverse-sorted tuples).
    void match(const std::vector<int>& cand,
               const std::vector<Box>& boxes,
               const std::vector<Track*>& tracks_,
               float thr,
               std::vector<std::pair<int, Track*>>& out,
               std::vector<int>& unusedCand,
               std::vector<Track*>& unusedTracks) const {
        if (cand.empty() || tracks_.empty()) {
            unusedCand = cand;
            unusedTracks = tracks_;
            return;
        }
        std::vector<std::tuple<float, size_t, size_t>> pairs;
        for (size_t r = 0; r < cand.size(); ++r)
            for (size_t c = 0; c < tracks_.size(); ++c) {
                const float v = iou(boxes[size_t(cand[r])], tracks_[c]->box());
                if (v >= thr) pairs.emplace_back(v, r, c);
            }
        std::sort(pairs.begin(), pairs.end(),
                  std::greater<std::tuple<float, size_t, size_t>>());
        std::set<size_t> usedD, usedT;
        for (const auto& p : pairs) {
            const size_t r = std::get<1>(p), c = std::get<2>(p);
            if (usedD.count(r) || usedT.count(c)) continue;
            usedD.insert(r);
            usedT.insert(c);
            out.emplace_back(cand[r], tracks_[c]);
        }
        for (size_t i = 0; i < cand.size(); ++i)
            if (!usedD.count(i)) unusedCand.push_back(cand[i]);
        for (size_t i = 0; i < tracks_.size(); ++i)
            if (!usedT.count(i)) unusedTracks.push_back(tracks_[i]);
    }
    Params p_;
    int next_ = 0;
};

// 1-D Kalman filter on the calibrated log-odds that one relation holds. No
// velocity term: a relation does not "become more true". The state persists
// when nothing is observed; predict only widens it.
struct RelationBelief {
    double x, P;
    RelationBelief(double z = 0, double P0 = 4.0) : x(z), P(P0) {}
    void predict(double q) { P += q; }
    void update(double z, double r, double pMin = .04) {
        const double K = P / (P + r);
        x += K * (z - x);
        P *= (1 - K);
        // Consecutive frames are nearly the same image, so measurements are
        // strongly correlated; flooring P keeps the belief able to change.
        P = std::max(P, pMin);
    }
    double prob() const {
        return 1.0 / (1.0 + std::exp(-std::max(-30.0, std::min(30.0, x))));
    }
};

inline double logit(double p, double eps = 1e-6) {
    p = std::min(std::max(p, eps), 1 - eps);
    return std::log(p / (1 - p));
}

struct Edge {
    int sub, obj, pred;
    float score;
    bool spatial;
    bool on = false;
    int onRun = 0, offRun = 0;
    float alpha = 0;
    int seen = 0;
    RelationBelief bel;
    float shown = 0;
};

// One frame of raw head output (borrowed). Logits are the RAW predicate and
// pair logits; the calibrated log-odds of (pair k, predicate v) is
// a * (pred[k, v] + pair[k]) + b.
struct RawFrame {
    const float* predLogits = nullptr;  // [K, V]
    const float* pairLogits = nullptr;  // [K]
    const std::int32_t* sub = nullptr;
    const std::int32_t* obj = nullptr;
    const std::int32_t* valid = nullptr;
    std::size_t K = 0, V = 0;
    float calA = 1.f, calB = 0.f;
    bool usable() const { return predLogits && pairLogits && K && V; }
    double z(std::size_t k, std::size_t v) const {
        return calA * (double(predLogits[k * V + v]) + pairLogits[k]) + calB;
    }
};

class EdgeBook {
public:
    EdgeBook(const Params& p, std::vector<std::string> predicates,
             std::vector<std::uint8_t> isSpatial)
        : p_(p),
          names_(std::move(predicates)),
          spatial_(std::move(isSpatial)) {
        spatial_.resize(names_.size(), 0);
    }
    std::vector<Edge> edges;

    // Promote the frame's best hypotheses, then update every live relation:
    // measured pairs get a Kalman update, unmeasurable ones (an endpoint was
    // not detected) only widen their belief. Falls back to EMA decay when
    // there is no raw frame or belief is off, exactly as upstream.
    void observe(const std::map<int, int>& det2track,
                 const std::vector<float>& detConfs,
                 const RawFrame& raw) {
        const bool useBelief = raw.usable() && p_.belief;
        std::vector<Birth> births;
        if (raw.usable()) births = birthList(det2track, detConfs, raw);
        if (!useBelief) {
            observeEma(det2track, births, raw);
            return;
        }
        std::map<int, int> track2det;
        for (const auto& kv : det2track) track2det[kv.second] = kv.first;
        std::map<std::pair<int, int>, std::size_t> kof;
        for (std::size_t k = 0; k < raw.K; ++k)
            if (!raw.valid || raw.valid[k])
                kof[{raw.sub[k], raw.obj[k]}] = k;
        for (const auto& b : births) {
            const auto s = det2track.find(raw.sub[b.k]);
            const auto o = det2track.find(raw.obj[b.k]);
            if (s == det2track.end() || o == det2track.end() ||
                s->second == o->second)
                continue;
            if (find(s->second, o->second, int(b.v)) < 0) {
                Edge e{s->second, o->second, int(b.v), b.score,
                       spatial_[b.v] != 0};
                e.bel = RelationBelief(logit(b.score));
                edges.push_back(e);
            }
        }
        const float offThr = p_.effectiveOffThr();
        std::vector<Edge> alive;
        for (auto& e : edges) {
            const auto ds = track2det.find(e.sub);
            const auto dobj = track2det.find(e.obj);
            bool measured = false;
            if (ds != track2det.end() && dobj != track2det.end()) {
                const auto k = kof.find({ds->second, dobj->second});
                if (k != kof.end()) {
                    // Straight off the raw matrices, NOT through the top-k:
                    // a pair that scored low never comes back through decode,
                    // so negative evidence could otherwise never be observed.
                    e.bel.update(raw.z(k->second, std::size_t(e.pred)),
                                 p_.relR);
                    ++e.seen;
                    measured = true;
                }
            }
            if (!measured) e.bel.predict(p_.relQ);
            e.score = float(e.bel.prob());
            if (e.score >= offThr) e.shown = e.score;
            if (e.bel.P <= p_.pMax) alive.push_back(e);  // else: unobserved too long
        }
        edges.swap(alive);
        hysteresis();
    }

    // Per rendered frame: fade in/out; drop edges whose endpoints vanished.
    void step(const std::set<int>& liveTracks) {
        std::vector<Edge> alive;
        for (auto& e : edges) {
            if (!liveTracks.count(e.sub) || !liveTracks.count(e.obj)) {
                e.on = false;
                e.alpha = std::max(0.f, e.alpha - p_.edgeFade);
                if (e.alpha > 0) alive.push_back(e);
                continue;
            }
            e.alpha = e.on ? std::min(1.f, e.alpha + p_.edgeFade)
                           : std::max(0.f, e.alpha - p_.edgeFade);
            if (!e.on && e.alpha <= 0 && e.score < .05f) continue;
            alive.push_back(e);
        }
        edges.swap(alive);
    }

    // What gets drawn. One edge per (object, predicate): a class-agnostic
    // segmenter splits a person into several regions that would each "play"
    // the same guitar. Then one per (pair, leading word): "playing" next to
    // "playing with" says the same thing twice. Interactions get first call
    // on the budget; spatial edges fill what is left, capped by maxSpatial.
    std::vector<Edge> visible() const {
        std::vector<Edge> vis;
        for (const auto& e : edges)
            if (e.alpha > .01f) vis.push_back(e);
        const auto strength = [](const Edge& e) {
            return std::max(e.shown, e.score);
        };
        const auto byStrength = [&](const Edge& a, const Edge& b) {
            return strength(a) > strength(b);
        };
        std::stable_sort(vis.begin(), vis.end(), byStrength);
        std::set<std::pair<int, int>> seenObj;
        std::vector<Edge> best;
        for (const auto& e : vis)
            if (seenObj.insert({e.obj, e.pred}).second) best.push_back(e);
        std::set<std::tuple<int, int, std::string>> seenHead;
        std::vector<Edge> head;
        for (const auto& e : best)
            if (seenHead.insert({e.sub, e.obj, firstWord(e.pred)}).second)
                head.push_back(e);
        std::stable_sort(head.begin(), head.end(), byStrength);
        std::vector<Edge> inter, spat;
        for (const auto& e : head) (e.spatial ? spat : inter).push_back(e);
        if (int(spat.size()) > p_.maxSpatial) spat.resize(size_t(p_.maxSpatial));
        const int room = std::max(0, p_.maxEdges - int(spat.size()));
        if (int(inter.size()) > room) inter.resize(size_t(room));
        inter.insert(inter.end(), spat.begin(), spat.end());
        return inter;
    }
    const std::string& name(int pred) const { return names_[size_t(pred)]; }

private:
    struct Birth {
        std::size_t k, v;
        float rank, score;
    };
    std::string firstWord(int pred) const {
        const std::string& n = names_[size_t(pred)];
        return n.substr(0, n.find(' '));
    }
    int find(int s, int o, int p) const {
        for (std::size_t i = 0; i < edges.size(); ++i)
            if (edges[i].sub == s && edges[i].obj == o && edges[i].pred == p)
                return int(i);
        return -1;
    }
    // upstream decode(): threshold 0 (render_video passes 0.0), rank by
    // score * conf(sub) * conf(obj), top birthTopk; maxPerPair == 1 keeps only
    // each pair's argmax predicate (any other value is unconstrained).
    std::vector<Birth> birthList(const std::map<int, int>&,
                                 const std::vector<float>& confs,
                                 const RawFrame& raw) const {
        const std::int32_t n = std::int32_t(confs.size());
        std::vector<Birth> all;
        for (std::size_t k = 0; k < raw.K; ++k) {
            if (raw.valid && !raw.valid[k]) continue;
            const std::int32_t s = raw.sub[k], o = raw.obj[k];
            if (s < 0 || o < 0 || s >= n || o >= n || s == o) continue;
            const float w = confs[size_t(s)] * confs[size_t(o)];
            std::size_t bestV = 0;
            double bestZ = -1e30;
            std::vector<Birth> row;
            for (std::size_t v = 0; v < raw.V && v < names_.size(); ++v) {
                const double z = raw.z(k, v);
                const float sc = float(1. / (1. + std::exp(-z)));
                row.push_back({k, v, sc * w, sc});
                if (z > bestZ) {
                    bestZ = z;
                    bestV = v;
                }
            }
            if (p_.maxPerPair == 1) {
                for (const auto& b : row)
                    if (b.v == bestV) all.push_back(b);
            } else {
                all.insert(all.end(), row.begin(), row.end());
            }
        }
        const std::size_t top =
                std::min(all.size(), std::size_t(std::max(0, p_.birthTopk)));
        std::partial_sort(all.begin(), all.begin() + std::ptrdiff_t(top),
                          all.end(), [](const Birth& a, const Birth& b) {
                              return a.rank > b.rank;
                          });
        all.resize(top);
        return all;
    }
    void observeEma(const std::map<int, int>& det2track,
                    const std::vector<Birth>& births,
                    const RawFrame& raw) {
        std::map<std::tuple<int, int, int>, float> fresh;
        for (const auto& b : births) {
            const auto s = det2track.find(raw.sub[b.k]);
            const auto o = det2track.find(raw.obj[b.k]);
            if (s == det2track.end() || o == det2track.end() ||
                s->second == o->second)
                continue;
            auto& v = fresh[{s->second, o->second, int(b.v)}];
            v = std::max(v, b.score);  // keep the stronger reading
        }
        const float offThr = p_.effectiveOffThr();
        for (const auto& kv : fresh) {
            const int s = std::get<0>(kv.first), o = std::get<1>(kv.first),
                      p = std::get<2>(kv.first);
            int i = find(s, o, p);
            if (i < 0) {
                edges.push_back({s, o, p, kv.second, spatial_[size_t(p)] != 0});
                i = int(edges.size()) - 1;
            } else {
                edges[size_t(i)].score = (1 - p_.scoreEma) *
                                                 edges[size_t(i)].score +
                                         p_.scoreEma * kv.second;
            }
            Edge& e = edges[size_t(i)];
            if (e.score >= offThr) e.shown = e.score;
            ++e.seen;
        }
        for (auto& e : edges)  // decay what was not re-observed
            if (!fresh.count({e.sub, e.obj, e.pred}))
                e.score = (1 - p_.scoreEma) * e.score;
        hysteresis();
    }
    // Run counters advance per OBSERVATION, on both score paths.
    void hysteresis() {
        const float offThr = p_.effectiveOffThr();
        for (auto& e : edges) {
            const float onThr = e.spatial ? p_.spatialOnThr : p_.onThr;
            if (e.score >= onThr) {
                ++e.onRun;
                e.offRun = 0;
            } else if (e.score < offThr) {
                ++e.offRun;
                e.onRun = 0;
            } else {
                e.onRun = e.offRun = 0;  // dead band: hold whatever we are
            }
            if (!e.on && e.onRun >= p_.onFrames)
                e.on = true;
            else if (e.on && e.offRun >= p_.offFrames)
                e.on = false;
        }
    }
    Params p_;
    std::vector<std::string> names_;
    std::vector<std::uint8_t> spatial_;
};

}  // namespace RelateTemporal
