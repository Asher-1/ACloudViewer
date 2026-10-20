// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// RelateAnything options of the qAnyLabeling workspace. The numbers are the
// two official operating points of RelateAnything-GGML (deploy/render_video.py
// and deploy/demo_webcam.py), the same ones the qRelateAnything plugin offers,
// so the workspace and the plugin report the same edges for the same boxes.
#include <QString>
#include <QStringList>

struct RelateOptions {
    enum class Profile { VideoReel = 0, ImageDemo = 1, Custom = 2 };
    // Edge selection (see RelateDecode::Params).
    float relationThreshold = .45f;  // interaction floor
    float spatialThreshold = .42f;   // spatial floor
    int maxRelations = 8;
    int maxSpatial = 3;
    bool decompose = true;  // two-graph decode (spatial / semantic)
    // Automatic detector stage, used when fewer than two rectangles are
    // selected. Detected boxes become real rectangles on the canvas.
    float detectConfidence = .20f;
    float detectIou = .5f;
    float maxBoxFraction = .85f;  // drop near-full-frame boxes
    int maxBoxes = 24;            // objects handed to the relation head
    // Custom predicate vocabulary, "a | b | c". Empty = the model's own list.
    QString predicates;
    // Detector GGUF file name of the object stage. Empty = the official demo
    // detector (yoloe-26s-seg-pf-f16.gguf).
    QString detector;
    Profile profile = Profile::VideoReel;

    bool operator==(const RelateOptions& o) const {
        return relationThreshold == o.relationThreshold &&
               spatialThreshold == o.spatialThreshold &&
               maxRelations == o.maxRelations && maxSpatial == o.maxSpatial &&
               decompose == o.decompose &&
               detectConfidence == o.detectConfidence &&
               detectIou == o.detectIou && maxBoxFraction == o.maxBoxFraction &&
               maxBoxes == o.maxBoxes && predicates == o.predicates &&
               detector == o.detector;
    }
    bool operator!=(const RelateOptions& o) const { return !(*this == o); }

    // deploy/render_video.py defaults (video reel) or deploy/demo_webcam.py
    // defaults (image demo). The predicate text is left to the caller.
    static RelateOptions forProfile(Profile p) {
        RelateOptions o;
        o.profile = p;
        if (p == Profile::ImageDemo) {
            o.relationThreshold = .5f;
            o.spatialThreshold = .5f;
            o.maxRelations = 10;
            o.maxSpatial = 10;
            o.decompose = false;
            o.detectConfidence = .25f;
            o.maxBoxFraction = 1.0f;
            o.maxBoxes = 32;
        }
        return o;
    }
    // Official confidence of a detector family (make_reel.py MODES: YOLOE
    // prompt-free 0.25, YOLO-World 0.15; otherwise the profile's own value).
    static float detectorConfidence(const QString& detector, Profile p) {
        if (p == Profile::ImageDemo) return .25f;
        if (detector.contains(QStringLiteral("-pf-"))) return .25f;
        if (detector.contains(QStringLiteral("-world"))) return .15f;
        return forProfile(p).detectConfidence;
    }
    // Which official profile these numbers are, or Custom. The detector's
    // own official confidence counts as matching.
    Profile matchedProfile() const {
        for (Profile p : {Profile::VideoReel, Profile::ImageDemo}) {
            RelateOptions ref = forProfile(p);
            ref.predicates = predicates;
            ref.detector = detector;
            if (*this == ref) return p;
            ref.detectConfidence = detectorConfidence(detector, p);
            if (*this == ref) return p;
        }
        return Profile::Custom;
    }
    // Predicates as the list the relation task expects, trimmed, no blanks.
    QStringList predicateList() const {
        QStringList out;
        for (const QString& name : predicates.split(QLatin1Char('|'),
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
                                                    Qt::SkipEmptyParts
#else
                                                    QString::SkipEmptyParts
#endif
                                                    )) {
            const QString trimmed = name.trimmed();
            if (!trimmed.isEmpty()) out << trimmed;
        }
        return out;
    }
};
