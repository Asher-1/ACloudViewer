// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QSpinBox>
#include <cstdio>

#include "RelateAnythingPresets.h"
#include "RelateDetectors.h"
#include "RelateOptions.h"
#include "RelateOptionsDialog.h"

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}
bool near(double a, double b) { return std::abs(a - b) < 1e-6; }
}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    // The two official operating points (render_video.py / demo_webcam.py)
    // are exactly what the qRelateAnything plugin offers.
    const RelateOptions reel =
            RelateOptions::forProfile(RelateOptions::Profile::VideoReel);
    const RelateOptions demo =
            RelateOptions::forProfile(RelateOptions::Profile::ImageDemo);
    const auto reelRef =
            RelatePresets::thresholds(RelatePresets::Profile::VideoReel);
    const auto demoRef =
            RelatePresets::thresholds(RelatePresets::Profile::ImageDemo);
    if (!check(near(reel.relationThreshold, reelRef.relation) &&
                       near(reel.spatialThreshold, reelRef.spatial) &&
                       reel.maxRelations == reelRef.maxRelations &&
                       reel.maxSpatial == reelRef.maxSpatial &&
                       reel.decompose == reelRef.decompose &&
                       near(reel.detectConfidence, reelRef.detConfidence) &&
                       near(reel.maxBoxFraction, reelRef.maxBoxFraction) &&
                       reel.maxBoxes == reelRef.maxBoxes,
               "video reel options must equal the qRelateAnything preset"))
        return 1;
    if (!check(near(demo.relationThreshold, demoRef.relation) &&
                       near(demo.spatialThreshold, demoRef.spatial) &&
                       demo.maxRelations == demoRef.maxRelations &&
                       demo.decompose == demoRef.decompose &&
                       near(demo.detectConfidence, demoRef.detConfidence) &&
                       near(demo.maxBoxFraction, demoRef.maxBoxFraction) &&
                       demo.maxBoxes == demoRef.maxBoxes,
               "image demo options must equal the qRelateAnything preset"))
        return 1;
    if (!check(reel.matchedProfile() == RelateOptions::Profile::VideoReel &&
                       demo.matchedProfile() ==
                               RelateOptions::Profile::ImageDemo,
               "a profile's own numbers must match that profile"))
        return 1;
    RelateOptions edited = reel;
    edited.relationThreshold = .6f;
    if (!check(edited.matchedProfile() == RelateOptions::Profile::Custom,
               "an edited number must read back as Custom"))
        return 1;

    // Predicate parsing: trimmed, no blanks.
    RelateOptions words;
    words.predicates = QStringLiteral(" riding |  | holding|next to ");
    if (!check(words.predicateList() ==
                       QStringList({"riding", "holding", "next to"}),
               "predicate list must be trimmed without blanks"))
        return 1;

    // The popup: loads, edits, flips to Custom, applies a profile.
    RelateOptionsDialog dialog(reel);
    dialog.show();
    auto* profile = dialog.findChild<QComboBox*>("RelateProfile");
    auto* relation =
            dialog.findChild<QDoubleSpinBox*>("RelateRelationThreshold");
    auto* maxRelations = dialog.findChild<QSpinBox*>("RelateMaxRelations");
    auto* vocabulary = dialog.findChild<QComboBox*>("RelateVocabulary");
    auto* predicates = dialog.findChild<QLineEdit*>("RelatePredicates");
    if (!check(profile && relation && maxRelations && vocabulary && predicates,
               "options dialog controls are missing"))
        return 1;
    if (!check(profile->currentData().toInt() ==
                               int(RelateOptions::Profile::VideoReel) &&
                       near(relation->value(), .45) &&
                       maxRelations->value() == 8,
               "the dialog must open on the passed options"))
        return 1;
    relation->setValue(.7);
    if (!check(profile->currentData().toInt() ==
                       int(RelateOptions::Profile::Custom),
               "editing a number must switch the operating point to Custom"))
        return 1;
    // Choosing the image-demo operating point refills every number but keeps
    // the predicate vocabulary.
    predicates->setText(QStringLiteral("riding | holding"));
    profile->setCurrentIndex(
            profile->findData(int(RelateOptions::Profile::ImageDemo)));
    QMetaObject::invokeMethod(profile, "activated",
                              Q_ARG(int, profile->currentIndex()));
    RelateOptions applied = dialog.options();
    if (!check(near(applied.relationThreshold, .5) &&
                       applied.maxRelations == 10 && !applied.decompose &&
                       applied.predicates ==
                               QStringLiteral("riding | holding") &&
                       applied.profile == RelateOptions::Profile::ImageDemo,
               "applying the image-demo point must refill the numbers and "
               "keep the predicates"))
        return 1;
    // Vocabulary presets write the predicate text; typing flips to Custom.
    vocabulary->setCurrentIndex(vocabulary->findData(0));
    QMetaObject::invokeMethod(vocabulary, "activated",
                              Q_ARG(int, vocabulary->currentIndex()));
    if (!check(dialog.options().predicateList() ==
                       RelatePresets::videoPredicates(),
               "the video vocabulary preset must write the official list"))
        return 1;
    vocabulary->setCurrentIndex(vocabulary->findData(2));
    QMetaObject::invokeMethod(vocabulary, "activated",
                              Q_ARG(int, vocabulary->currentIndex()));
    if (!check(dialog.options().predicates.isEmpty(),
               "the full-vocabulary preset must clear the predicates"))
        return 1;
    // The detector choice: same catalog as the standalone plugin, the
    // official default first-class, and each family keeps its own official
    // confidence (YOLO-World 0.15, prompt-free 0.25).
    const auto detectors = RelateDetectors::catalog();
    if (!check(!detectors.isEmpty() &&
                       RelateDetectors::info(RelateDetectors::defaultFilename())
                               .valid(),
               "the detector catalog must contain the official default"))
        return 1;
    RelateDetectors::Info world;
    for (const auto& d : detectors)
        if (d.world && d.vocabulary) {
            world = d;
            break;
        }
    RelateOptionsDialog picker(reel, detectors);
    auto* detector = picker.findChild<QComboBox*>("RelateDetector");
    auto* confidence =
            picker.findChild<QDoubleSpinBox*>("RelateDetectConfidence");
    if (!check(detector && confidence && detector->count() == detectors.size(),
               "the detector combo must list the whole catalog"))
        return 1;
    if (!check(picker.options().detector.isEmpty(),
               "the default detector is stored as empty"))
        return 1;
    if (world.valid()) {
        detector->setCurrentIndex(detector->findData(world.filename));
        QMetaObject::invokeMethod(detector, "activated",
                                  Q_ARG(int, detector->currentIndex()));
        if (!check(near(confidence->value(), .15) &&
                           picker.options().detector == world.filename &&
                           picker.options().matchedProfile() ==
                                   RelateOptions::Profile::VideoReel,
                   "YOLO-World must use its official 0.15 confidence and "
                   "stay on the video-reel operating point"))
            return 1;
        RelateOptionsDialog again(picker.options(), detectors);
        if (!check(again.options().detector == world.filename,
                   "a reopened dialog must keep the chosen detector"))
            return 1;
    }
    // A reopened dialog shows what was saved.
    RelateOptionsDialog reopened(dialog.options());
    if (!check(reopened.options() == dialog.options(),
               "reopening the dialog must reproduce the saved options"))
        return 1;
    return 0;
}
