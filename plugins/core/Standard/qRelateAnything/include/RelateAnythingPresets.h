// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <QString>
#include <QStringList>

// Official RelateAnything operating points (github.com/Maelic/RelateAnything,
// AGPL-3.0; lists and numbers reproduced as data).
namespace RelatePresets {

// deploy/render_video.py: the curated predicate vocabulary behind the project
// reel (interaction first, spatial kept few so they do not crowd the top-k).
inline QStringList videoPredicates() {
    return {"riding",
            "holding",
            "carrying",
            "playing",
            "using",
            "wearing",
            "sitting on",
            "sitting at",
            "standing on",
            "looking at",
            "watching",
            "eating",
            "eating from",
            "drinking from",
            "cutting",
            "pushing",
            "pulling",
            "climbing",
            "talking to",
            "walking with",
            "leaning against",
            "reading",
            "playing with",
            "feeding",
            "petting",
            "hitting",
            "kicking",
            "driving",
            "steering",
            "pedaling",
            "lying on",
            "hugging",
            "embracing",
            "serving",
            "preparing",
            "operating",
            "typing on",
            "photographing",
            "pointing at",
            "shaking hands with",
            "holding hands with",
            "dancing with",
            "gripping",
            "reaching for",
            "swinging",
            "leading",
            "following",
            "walking on",
            "jumping over",
            "resting on",
            "hanging from",
            "attached to",
            "touching",
            "on",
            "inside",
            "on top of",
            "in front of",
            "beside",
            "behind",
            "above",
            "below",
            "next to",
            "near",
            "under",
            "over"};
}

// deploy/demo_webcam.py preset "all" (interaction + spatial).
inline QStringList webcamPredicates() {
    return {"wearing",      "riding",         "playing",
            "sitting on",   "sitting at",     "holding",
            "sitting in",   "looking at",     "using",
            "watching",     "standing on",    "carrying",
            "talking to",   "smiling at",     "standing beside",
            "walking past", "posing with",    "leaning against",
            "part of",      "resting on",     "covering",
            "inside",       "contained in",   "hanging from",
            "surrounding",  "attached to",    "on",
            "on top of",    "in front of",    "behind",
            "beside",       "to the left of", "to the right of",
            "above",        "below"};
}

inline QString join(const QStringList& names) { return names.join(" | "); }

enum class Profile { VideoReel = 0, ImageDemo = 1, Custom = 2 };

// Threshold operating points. VideoReel = deploy/render_video.py defaults,
// ImageDemo = deploy/demo_webcam.py defaults (single graph, flat floor).
struct Thresholds {
    double detConfidence, detIou, relation, spatial, maxBoxFraction;
    int maxBoxes, maxRelations, maxSpatial;
    bool decompose;
};
inline Thresholds thresholds(Profile p) {
    if (p == Profile::ImageDemo)
        return {0.25, 0.5, 0.5, 0.5, 1.0, 32, 10, 10, false};
    return {0.20, 0.5, 0.45, 0.42, 0.85, 24, 8, 3, true};
}

}  // namespace RelatePresets
