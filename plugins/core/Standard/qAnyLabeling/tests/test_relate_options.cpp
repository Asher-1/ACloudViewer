// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QApplication>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QSpinBox>
#include <cmath>
#include <memory>

#include "RelateAnythingPresets.h"
#include "RelateDetectors.h"
#include "RelateOptions.h"
#include "RelateOptionsDialog.h"

namespace {
class RelateProfileTest
    : public ::testing::TestWithParam<RelateOptions::Profile> {};

TEST_P(RelateProfileTest, MatchesOfficialStandalonePresetAndRecognizesItself) {
    const auto profile = GetParam();
    const auto options = RelateOptions::forProfile(profile);
    const auto reference = RelatePresets::thresholds(
            profile == RelateOptions::Profile::VideoReel
                    ? RelatePresets::Profile::VideoReel
                    : RelatePresets::Profile::ImageDemo);
    // Keep the original strict absolute tolerance for float/double comparisons.
    EXPECT_LT(std::abs(options.relationThreshold - reference.relation), 1e-6);
    EXPECT_LT(std::abs(options.spatialThreshold - reference.spatial), 1e-6);
    EXPECT_EQ(options.maxRelations, reference.maxRelations);
    EXPECT_EQ(options.maxSpatial, reference.maxSpatial);
    EXPECT_EQ(options.decompose, reference.decompose);
    EXPECT_LT(std::abs(options.detectConfidence - reference.detConfidence),
              1e-6);
    EXPECT_LT(std::abs(options.maxBoxFraction - reference.maxBoxFraction),
              1e-6);
    EXPECT_EQ(options.maxBoxes, reference.maxBoxes);
    EXPECT_EQ(options.matchedProfile(), profile);
}

INSTANTIATE_TEST_SUITE_P(
        OfficialOperatingPoints,
        RelateProfileTest,
        ::testing::Values(RelateOptions::Profile::VideoReel,
                          RelateOptions::Profile::ImageDemo),
        [](const ::testing::TestParamInfo<RelateOptions::Profile>& info) {
            return info.param == RelateOptions::Profile::VideoReel
                           ? "VideoReel"
                           : "ImageDemo";
        });

TEST(RelateOptionsTest, EditingAProfileNumberMakesItCustom) {
    auto options = RelateOptions::forProfile(RelateOptions::Profile::VideoReel);
    options.relationThreshold = .6f;
    EXPECT_EQ(options.matchedProfile(), RelateOptions::Profile::Custom);
}

TEST(RelateOptionsTest, PredicateListTrimsWordsAndDropsBlankEntries) {
    RelateOptions options;
    options.predicates = QStringLiteral(" riding |  | holding|next to ");
    EXPECT_EQ(options.predicateList(),
              (QStringList{"riding", "holding", "next to"}));
}

TEST(RelateOptionsTest, DetectorCatalogContainsOfficialDefault) {
    const auto detectors = RelateDetectors::catalog();
    EXPECT_FALSE(detectors.isEmpty());
    EXPECT_TRUE(
            RelateDetectors::info(RelateDetectors::defaultFilename()).valid());
}

class RelateOptionsDialogTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        static int argc = 1;
        static char name[] = "test_qanylabeling_relate_options";
        static char* argv[] = {name, nullptr};
        static QApplication app(argc, argv);
    }

    void SetUp() override {
        reel = RelateOptions::forProfile(RelateOptions::Profile::VideoReel);
        dialog = std::make_unique<RelateOptionsDialog>(reel);
        dialog->show();
        profile = dialog->findChild<QComboBox*>("RelateProfile");
        relation =
                dialog->findChild<QDoubleSpinBox*>("RelateRelationThreshold");
        maxRelations = dialog->findChild<QSpinBox*>("RelateMaxRelations");
        vocabulary = dialog->findChild<QComboBox*>("RelateVocabulary");
        predicates = dialog->findChild<QLineEdit*>("RelatePredicates");
        ASSERT_NE(profile, nullptr);
        ASSERT_NE(relation, nullptr);
        ASSERT_NE(maxRelations, nullptr);
        ASSERT_NE(vocabulary, nullptr);
        ASSERT_NE(predicates, nullptr);
    }

    RelateOptions reel;
    std::unique_ptr<RelateOptionsDialog> dialog;
    QComboBox* profile = nullptr;
    QDoubleSpinBox* relation = nullptr;
    QSpinBox* maxRelations = nullptr;
    QComboBox* vocabulary = nullptr;
    QLineEdit* predicates = nullptr;
};

TEST_F(RelateOptionsDialogTest, OpensWithPassedOperatingPoint) {
    EXPECT_EQ(profile->currentData().toInt(),
              int(RelateOptions::Profile::VideoReel));
    EXPECT_LT(std::abs(relation->value() - .45), 1e-6);
    EXPECT_EQ(maxRelations->value(), 8);
    EXPECT_EQ(dialog->options(), reel);
}

TEST_F(RelateOptionsDialogTest, EditingANumberSwitchesToCustom) {
    relation->setValue(.7);
    EXPECT_EQ(profile->currentData().toInt(),
              int(RelateOptions::Profile::Custom));
    EXPECT_LT(std::abs(dialog->options().relationThreshold - .7), 1e-6);
}

TEST_F(RelateOptionsDialogTest,
       ApplyingImageDemoRefillsNumbersAndKeepsPredicates) {
    relation->setValue(.7);
    predicates->setText(QStringLiteral("riding | holding"));
    const int imageDemo =
            profile->findData(int(RelateOptions::Profile::ImageDemo));
    ASSERT_GE(imageDemo, 0);
    profile->setCurrentIndex(imageDemo);
    ASSERT_TRUE(QMetaObject::invokeMethod(profile, "activated",
                                          Q_ARG(int, imageDemo)));
    const auto applied = dialog->options();
    EXPECT_LT(std::abs(applied.relationThreshold - .5), 1e-6);
    EXPECT_EQ(applied.maxRelations, 10);
    EXPECT_FALSE(applied.decompose);
    EXPECT_EQ(applied.predicates, QStringLiteral("riding | holding"));
    EXPECT_EQ(applied.profile, RelateOptions::Profile::ImageDemo);
    auto expected =
            RelateOptions::forProfile(RelateOptions::Profile::ImageDemo);
    expected.predicates = applied.predicates;
    EXPECT_EQ(applied, expected);
}

TEST_F(RelateOptionsDialogTest, VideoVocabularyWritesOfficialPredicates) {
    const int video = vocabulary->findData(0);
    ASSERT_GE(video, 0);
    vocabulary->setCurrentIndex(video);
    ASSERT_TRUE(QMetaObject::invokeMethod(vocabulary, "activated",
                                          Q_ARG(int, video)));
    EXPECT_EQ(dialog->options().predicateList(),
              RelatePresets::videoPredicates());
}

TEST_F(RelateOptionsDialogTest, FullVocabularyClearsPredicates) {
    predicates->setText(QStringLiteral("riding | holding"));
    const int full = vocabulary->findData(2);
    ASSERT_GE(full, 0);
    vocabulary->setCurrentIndex(full);
    ASSERT_TRUE(QMetaObject::invokeMethod(vocabulary, "activated",
                                          Q_ARG(int, full)));
    EXPECT_TRUE(dialog->options().predicates.isEmpty());
}

TEST_F(RelateOptionsDialogTest, ReopeningPreservesEditedOptions) {
    relation->setValue(.7);
    predicates->setText(QStringLiteral("riding | holding"));
    const auto saved = dialog->options();
    RelateOptionsDialog reopened(saved);
    EXPECT_EQ(reopened.options(), saved);
}

TEST_F(RelateOptionsDialogTest,
       DetectorPickerListsCatalogAndStoresDefaultAsEmpty) {
    const auto detectors = RelateDetectors::catalog();
    RelateOptionsDialog picker(reel, detectors);
    auto* detector = picker.findChild<QComboBox*>("RelateDetector");
    auto* confidence =
            picker.findChild<QDoubleSpinBox*>("RelateDetectConfidence");
    ASSERT_NE(detector, nullptr);
    ASSERT_NE(confidence, nullptr);
    EXPECT_EQ(detector->count(), detectors.size());
    EXPECT_TRUE(picker.options().detector.isEmpty());
}

TEST_F(RelateOptionsDialogTest,
       WorldDetectorKeepsOfficialConfidenceAndSurvivesReopen) {
    const auto detectors = RelateDetectors::catalog();
    RelateDetectors::Info world;
    for (const auto& detector : detectors) {
        if (detector.world && detector.vocabulary) {
            world = detector;
            break;
        }
    }
    ASSERT_TRUE(world.valid())
            << "Catalog must expose a vocabulary-based YOLO-World detector";
    RelateOptionsDialog picker(reel, detectors);
    auto* detector = picker.findChild<QComboBox*>("RelateDetector");
    auto* confidence =
            picker.findChild<QDoubleSpinBox*>("RelateDetectConfidence");
    ASSERT_NE(detector, nullptr);
    ASSERT_NE(confidence, nullptr);
    const int worldIndex = detector->findData(world.filename);
    ASSERT_GE(worldIndex, 0);
    detector->setCurrentIndex(worldIndex);
    ASSERT_TRUE(QMetaObject::invokeMethod(detector, "activated",
                                          Q_ARG(int, worldIndex)));
    EXPECT_LT(std::abs(confidence->value() - .15), 1e-6);
    EXPECT_EQ(picker.options().detector, world.filename);
    EXPECT_EQ(picker.options().matchedProfile(),
              RelateOptions::Profile::VideoReel);
    RelateOptionsDialog reopened(picker.options(), detectors);
    EXPECT_EQ(reopened.options().detector, world.filename);
}
}  // namespace
