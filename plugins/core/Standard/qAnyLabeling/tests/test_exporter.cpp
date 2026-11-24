// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QBuffer>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSet>
#include <QTemporaryDir>
#include <QVector>

#if defined(QANYLABELING_HAS_OPENCV_MASK)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "AnnotationExporter.h"
#include "AnnotationImporter.h"
#include "AnnotationMask.h"
#include "AnnotationMot.h"
#include "AnnotationOdvg.h"
#include "AnnotationPpocr.h"
#include "AnnotationVlm.h"

namespace {
class AnnotationExporterTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        static int argc = 1;
        static char name[] = "test_qanylabeling_exporter";
        static char* argv[] = {name, nullptr};
        static QCoreApplication app(argc, argv);
    }

    void SetUp() override {
        ocrShape = {
                {QStringLiteral("label"), QStringLiteral("text")},
                {QStringLiteral("description"), QStringLiteral("hello")},
                {QStringLiteral("shape_type"), QStringLiteral("quadrilateral")},
                {QStringLiteral("group_id"), 3},
                {QStringLiteral("difficult"), true},
                {QStringLiteral("kie_linking"), QJsonArray{QJsonArray{3, 4}}},
                {QStringLiteral("points"),
                 QJsonArray{QJsonArray{1, 2}, QJsonArray{30, 2},
                            QJsonArray{30, 10}, QJsonArray{1, 10}}}};
        ocrDocument = {{QStringLiteral("shapes"), QJsonArray{ocrShape}}};
        box = {{QStringLiteral("label"), QStringLiteral("bus")},
               {QStringLiteral("shape_type"), QStringLiteral("rectangle")},
               {QStringLiteral("points"),
                QJsonArray{QJsonArray{10, 20}, QJsonArray{50, 60}}}};
        polygon = {{QStringLiteral("label"), QStringLiteral("person")},
                   {QStringLiteral("shape_type"), QStringLiteral("polygon")},
                   {QStringLiteral("points"),
                    QJsonArray{QJsonArray{-10, 0}, QJsonArray{50, 0},
                               QJsonArray{50, 100}}}};
        oriented = {{QStringLiteral("label"), QStringLiteral("bus")},
                    {QStringLiteral("shape_type"), QStringLiteral("rotation")},
                    {QStringLiteral("points"),
                     QJsonArray{QJsonArray{-10, 20}, QJsonArray{50, 20},
                                QJsonArray{50, 60}, QJsonArray{-10, 60}}}};
        document = {
                {QStringLiteral("imageWidth"), 100},
                {QStringLiteral("imageHeight"), 100},
                {QStringLiteral("shapes"), QJsonArray{box, polygon, oriented}}};
        classes = QStringList{QStringLiteral("person"), QStringLiteral("bus")};
        poseConfig = parsePoseConfig(
                "has_visible: true\nclasses:\n  person:\n    - nose\n    - "
                "eye\n"
                "  dog:\n    - nose\n    - ear\n    - paw\n");
        poseBox = box;
        poseBox.insert(QStringLiteral("label"), QStringLiteral("person"));
        poseBox.insert(QStringLiteral("group_id"), 4);
        nose = {{QStringLiteral("label"), QStringLiteral("nose")},
                {QStringLiteral("shape_type"), QStringLiteral("point")},
                {QStringLiteral("group_id"), 4},
                {QStringLiteral("points"), QJsonArray{QJsonArray{20.9, 30.2}}}};
        cat = {{QStringLiteral("label"), QStringLiteral("cat")},
               {QStringLiteral("shape_type"), QStringLiteral("polygon")},
               {QStringLiteral("points"),
                QJsonArray{QJsonArray{1, 1}, QJsonArray{14, 1},
                           QJsonArray{14, 14}, QJsonArray{1, 14}}}};
        dog = {{QStringLiteral("label"), QStringLiteral("dog")},
               {QStringLiteral("shape_type"), QStringLiteral("polygon")},
               {QStringLiteral("points"),
                QJsonArray{QJsonArray{5, 5}, QJsonArray{10, 5},
                           QJsonArray{10, 10}, QJsonArray{5, 10}}}};
        maskDocument = {{QStringLiteral("imageWidth"), 16},
                        {QStringLiteral("imageHeight"), 16},
                        {QStringLiteral("shapes"), QJsonArray{cat, dog}}};
        rgbMap = {
                {QStringLiteral("type"), QStringLiteral("rgb")},
                {QStringLiteral("colors"),
                 QJsonObject{{QStringLiteral("cat"), QJsonArray{85, 0, 255}},
                             {QStringLiteral("dog"), QJsonArray{213, 42, 54}}}},
                {QStringLiteral("label_priority"),
                 QJsonObject{{QStringLiteral("cat"), 2},
                             {QStringLiteral("dog"), 1}}}};
    }
    QJsonObject ocrShape, ocrDocument;
    QJsonObject box, polygon, oriented, document;
    QStringList classes;
    PoseConfig poseConfig;
    QJsonObject poseBox, nose;
    QJsonObject cat, dog, maskDocument, rgbMap;
};

class PpocrRoundTripTest : public AnnotationExporterTest,
                           public ::testing::WithParamInterface<PpocrMode> {};

TEST_P(PpocrRoundTripTest, PreservesAnnotationAndKieLinks) {
    const auto mode = GetParam();
    const auto exported =
            convertPpocr({{QStringLiteral("ocr.png"), ocrDocument}}, mode);
    ASSERT_TRUE(exported.error.isEmpty()) << exported.error.toStdString();
    EXPECT_EQ(exported.count, 1);
    const auto imported = importPpocr(exported.text, mode);
    ASSERT_TRUE(imported.error.isEmpty()) << imported.error.toStdString();
    EXPECT_EQ(imported.count, 1);
    if (mode == PpocrMode::Kie) {
        const auto shapes = imported.documents.value(QStringLiteral("ocr.png"))
                                    .value(QStringLiteral("shapes"))
                                    .toArray();
        ASSERT_EQ(shapes.size(), 1);
        EXPECT_EQ(shapes.first()
                          .toObject()
                          .value(QStringLiteral("kie_linking"))
                          .toArray()
                          .size(),
                  1);
    }
}

INSTANTIATE_TEST_SUITE_P(Modes,
                         PpocrRoundTripTest,
                         ::testing::Values(PpocrMode::Rec, PpocrMode::Kie),
                         [](const ::testing::TestParamInfo<PpocrMode>& info) {
                             return info.param == PpocrMode::Rec ? "Rec"
                                                                 : "Kie";
                         });

TEST_F(AnnotationExporterTest, PpocrRejectsMalformedRecognitionInput) {
    EXPECT_FALSE(importPpocr("bad\n", PpocrMode::Rec).error.isEmpty());
}

TEST_F(AnnotationExporterTest,
       PpocrKiePreservesDirectedLinksNullIdsAndIntegerCoordinates) {
    // KIE source contract: integer coordinates truncate toward zero, absent
    // IDs default to 0, explicit null IDs and directed links survive export.
    QJsonObject question = ocrShape;
    question["label"] = QStringLiteral("question");
    question["description"] = QStringLiteral("姓名？");
    question["points"] =
            QJsonArray{QJsonArray{-1.9, 2.9}, QJsonArray{30.8, 2.9},
                       QJsonArray{30.8, 10.2}, QJsonArray{-1.9, 10.2}};
    QJsonObject answer = ocrShape;
    answer["label"] = QStringLiteral("answer");
    answer["description"] = QStringLiteral("张三");
    answer["group_id"] = 4;
    QJsonObject nullId = answer;
    nullId["group_id"] = QJsonValue::Null;
    nullId["kie_linking"] = QJsonArray();
    QJsonObject missingId = nullId;
    missingId.remove("group_id");
    const auto linkedKie = convertPpocr(
            {{QStringLiteral("文档.png"),
              QJsonObject{{"shapes",
                           QJsonArray{question, answer, nullId, missingId}}}}},
            PpocrMode::Kie);
    const auto exportedEntities =
            QJsonDocument::fromJson(
                    linkedKie.text.mid(linkedKie.text.indexOf('\t') + 1)
                            .trimmed())
                    .array();
    const auto linkedImport = importPpocr(linkedKie.text, PpocrMode::Kie);
    const auto reexport = convertPpocr(
            {{QStringLiteral("文档.png"),
              linkedImport.documents.value(QStringLiteral("文档.png"))}},
            PpocrMode::Kie);
    ASSERT_TRUE(linkedKie.error.isEmpty()) << linkedKie.error.toStdString();
    ASSERT_TRUE(linkedImport.error.isEmpty())
            << linkedImport.error.toStdString();
    ASSERT_EQ(linkedKie.count, 4);
    ASSERT_EQ(linkedImport.count, 4);
    ASSERT_EQ(exportedEntities.size(), 4);
    ASSERT_EQ(exportedEntities[0].toObject()["points"].toArray()[0].toArray(),
              (QJsonArray{-1, 2}));
    ASSERT_EQ(exportedEntities[0].toObject()["transcription"],
              QStringLiteral("姓名？"));
    ASSERT_EQ(exportedEntities[1].toObject()["linking"].toArray(),
              (QJsonArray{QJsonArray{3, 4}}));
    ASSERT_TRUE(exportedEntities[2].toObject()["id"].isNull());
    ASSERT_EQ(exportedEntities[3].toObject()["id"].toInt(-1), 0);
    ASSERT_EQ(reexport.text, linkedKie.text);
}

TEST_F(AnnotationExporterTest,
       PpocrRecDatasetReplacementRemovesStaleCropsAndLabels) {
    QTemporaryDir ppocrDirectory;
    QImage ppocrImage(40, 20, QImage::Format_RGB32);
    ppocrImage.fill(Qt::white);
    const QString ppocrImagePath =
            ppocrDirectory.filePath(QStringLiteral("ocr.png"));
    const QString ppocrOutput =
            ppocrDirectory.filePath(QStringLiteral("dataset"));
    ASSERT_TRUE(ppocrDirectory.isValid());
    ASSERT_TRUE(ppocrImage.save(ppocrImagePath));
    const PpocrRecExport ppocrExport =
            exportPpocrRecDataset({{ppocrImagePath, ocrDocument}}, ppocrOutput);
    QFile ppocrLabels(QDir(ppocrOutput).filePath(QStringLiteral("Label.txt")));
    QFile ppocrGt(QDir(ppocrOutput).filePath(QStringLiteral("rec_gt.txt")));
    const QString cropPath =
            QDir(ppocrOutput)
                    .filePath(QStringLiteral("crop_img/ocr_crop_0.jpg"));
    ASSERT_TRUE(ppocrExport.error.isEmpty()) << ppocrExport.error.toStdString();
    ASSERT_EQ(ppocrExport.annotationCount, 1);
    ASSERT_EQ(ppocrExport.cropCount, 1);
    ASSERT_TRUE(ppocrLabels.open(QIODevice::ReadOnly));
    ASSERT_TRUE(ppocrGt.open(QIODevice::ReadOnly));
    ASSERT_FALSE(QImage(cropPath).isNull());
    ASSERT_TRUE(ppocrLabels.readAll().contains("ocr.png\t"));
    ASSERT_TRUE(ppocrGt.readAll().contains("crop_img/ocr_crop_0.jpg\thello\n"));
    const PpocrRecExport ppocrEmpty = exportPpocrRecDataset({}, ppocrOutput);
    ASSERT_TRUE(ppocrEmpty.error.isEmpty()) << ppocrEmpty.error.toStdString();
    ASSERT_FALSE(QFile::exists(cropPath));
    ASSERT_EQ(QFileInfo(ppocrLabels.fileName()).size(), 0);
    ASSERT_EQ(QFileInfo(ppocrGt.fileName()).size(), 0);
}

TEST_F(AnnotationExporterTest,
       PpocrRectangleCropPreservesOrientationAndUnicodeTranscription) {
    QTemporaryDir ppocrDirectory;
    ASSERT_TRUE(ppocrDirectory.isValid());
    const QString ppocrImagePath =
            ppocrDirectory.filePath(QStringLiteral("ocr.png"));
    const QString ppocrOutput =
            ppocrDirectory.filePath(QStringLiteral("dataset"));
    const QString cropPath =
            QDir(ppocrOutput)
                    .filePath(QStringLiteral("crop_img/ocr_crop_0.jpg"));
    QImage orientedImage(40, 20, QImage::Format_RGB32);
    orientedImage.fill(Qt::red);
    for (int y = 0; y < orientedImage.height(); ++y)
        for (int x = 20; x < orientedImage.width(); ++x)
            orientedImage.setPixelColor(x, y, Qt::blue);
    ASSERT_TRUE(orientedImage.save(ppocrImagePath));
    const QJsonObject twoPointShape{
            {QStringLiteral("shape_type"), QStringLiteral("rectangle")},
            {QStringLiteral("description"), QStringLiteral("中文")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{0, 0}, QJsonArray{39, 19}}}};
    const PpocrRecExport orientedExport = exportPpocrRecDataset(
            {{ppocrImagePath, QJsonObject{{QStringLiteral("shapes"),
                                           QJsonArray{twoPointShape}}}}},
            ppocrOutput);
    const QImage orientedCrop(cropPath);
    QFile orientedGt(QDir(ppocrOutput).filePath(QStringLiteral("rec_gt.txt")));
    ASSERT_TRUE(orientedExport.error.isEmpty())
            << orientedExport.error.toStdString();
    ASSERT_EQ(orientedExport.cropCount, 1);
    ASSERT_FALSE(orientedCrop.isNull());
    ASSERT_GE(orientedCrop.width(), orientedCrop.height());
    ASSERT_GE(orientedCrop.pixelColor(2, orientedCrop.height() / 2).red(), 200);
    ASSERT_GE(orientedCrop
                      .pixelColor(orientedCrop.width() - 3,
                                  orientedCrop.height() / 2)
                      .blue(),
              200);
    ASSERT_TRUE(orientedGt.open(QIODevice::ReadOnly));
    ASSERT_TRUE(orientedGt.readAll().contains(QStringLiteral("中文").toUtf8()));
}

TEST_F(AnnotationExporterTest, YoloBoxesNormalizeOnlyRectangleCoordinates) {
    const auto boxes = convertYolo(document, classes, YoloFormat::Boxes);
    ASSERT_EQ(boxes.count, 1);
    EXPECT_EQ(boxes.text, "1 0.300000 0.400000 0.400000 0.400000\n");
}

TEST_F(AnnotationExporterTest,
       YoloSegmentsClampPolygonCoordinatesToImageBounds) {
    const auto segments = convertYolo(document, classes, YoloFormat::Segments);
    ASSERT_EQ(segments.count, 1);
    EXPECT_EQ(segments.text,
              "0 0.000000 0.000000 0.500000 0.000000 0.500000 1.000000\n");
}

TEST_F(AnnotationExporterTest,
       YoloOrientedBoxesPreserveOrSkipOutOfBoundsCoordinates) {
    const auto obb = convertYolo(document, classes, YoloFormat::OrientedBoxes);
    ASSERT_EQ(obb.count, 1);
    EXPECT_EQ(obb.text,
              "1 -0.100000 0.200000 0.500000 0.200000 0.500000 0.600000 "
              "-0.100000 0.600000\n");
    const auto skipped =
            convertYolo(document, classes, YoloFormat::OrientedBoxes, false);
    EXPECT_EQ(skipped.count, 0);
    EXPECT_TRUE(skipped.text.isEmpty());
}

TEST_F(AnnotationExporterTest, DotaSkipsOutOfBoundsOrientedBoxes) {
    EXPECT_EQ(convertDota(document).count, 0);
}

TEST_F(AnnotationExporterTest,
       VocPolygonModeIncludesPolygonsAndClampsCoordinates) {
    const auto boxes = convertVoc(document, QStringLiteral("image.png"),
                                  QStringLiteral("labels"), false);
    const auto polygons = convertVoc(document, QStringLiteral("image.png"),
                                     QStringLiteral("labels"), true);
    EXPECT_EQ(boxes.count, 1);
    EXPECT_EQ(polygons.count, 2);
    EXPECT_TRUE(polygons.text.contains("<x1>0</x1>"));
    EXPECT_TRUE(polygons.text.contains("<filename>image.png</filename>"));
}

TEST_F(AnnotationExporterTest,
       VocRoundTripPreservesShapeTypesAndRejectsMalformedXml) {
    const auto vocBoxes = convertVoc(document, QStringLiteral("image.png"),
                                     QStringLiteral("labels"), false);
    const auto vocPolygons = convertVoc(document, QStringLiteral("image.png"),
                                        QStringLiteral("labels"), true);
    const AnnotationImport importedVocBox =
            importVoc(vocBoxes.text, QStringLiteral("image.png"), false);
    const AnnotationImport importedVocPolygons =
            importVoc(vocPolygons.text, QStringLiteral("image.png"), true);
    const QJsonArray vocShapes =
            importedVocPolygons.document.value(QStringLiteral("shapes"))
                    .toArray();
    ASSERT_TRUE(importedVocBox.error.isEmpty())
            << importedVocBox.error.toStdString();
    ASSERT_EQ(importedVocBox.count, 1);
    ASSERT_TRUE(importedVocPolygons.error.isEmpty())
            << importedVocPolygons.error.toStdString();
    ASSERT_EQ(importedVocPolygons.count, 2);
    ASSERT_EQ(vocShapes.size(), 2);
    ASSERT_EQ(vocShapes[0]
                      .toObject()
                      .value(QStringLiteral("shape_type"))
                      .toString(),
              QStringLiteral("rectangle"));
    ASSERT_EQ(vocShapes[1]
                      .toObject()
                      .value(QStringLiteral("shape_type"))
                      .toString(),
              QStringLiteral("polygon"));
    ASSERT_FALSE(importVoc("<annotation><object></annotation>",
                           QStringLiteral("image.png"), false)
                         .error.isEmpty());
}

class YoloRoundTripTest : public AnnotationExporterTest,
                          public ::testing::WithParamInterface<YoloFormat> {};

TEST_P(YoloRoundTripTest, PreservesShapeTypeNullGroupAndImagePath) {
    const auto format = GetParam();
    const YoloConversion exported = convertYolo(document, classes, format);
    const YoloImport imported =
            importYolo(exported.text, classes, format,
                       QStringLiteral("image.png"), QSize(100, 100));
    const QJsonArray shapes =
            imported.document.value(QStringLiteral("shapes")).toArray();
    const QString type =
            format == YoloFormat::Boxes      ? QStringLiteral("rectangle")
            : format == YoloFormat::Segments ? QStringLiteral("polygon")
                                             : QStringLiteral("rotation");
    ASSERT_TRUE(imported.error.isEmpty()) << imported.error.toStdString();
    ASSERT_EQ(imported.count, 1);
    ASSERT_EQ(shapes.size(), 1);
    ASSERT_EQ(
            shapes[0].toObject().value(QStringLiteral("shape_type")).toString(),
            type);
    ASSERT_EQ(shapes[0].toObject().value(QStringLiteral("group_id")).type(),
              QJsonValue::Null);
    ASSERT_EQ(imported.document.value(QStringLiteral("imagePath")).toString(),
              QStringLiteral("image.png"));
}

INSTANTIATE_TEST_SUITE_P(Formats,
                         YoloRoundTripTest,
                         ::testing::Values(YoloFormat::Boxes,
                                           YoloFormat::Segments,
                                           YoloFormat::OrientedBoxes),
                         [](const ::testing::TestParamInfo<YoloFormat>& info) {
                             switch (info.param) {
                                 case YoloFormat::Boxes:
                                     return "Boxes";
                                 case YoloFormat::Segments:
                                     return "Segments";
                                 default:
                                     return "OrientedBoxes";
                             }
                         });

TEST_F(AnnotationExporterTest,
       YoloRejectsUnknownClassesNonfiniteNumbersAndInvalidSegments) {
    ASSERT_FALSE(importYolo("9 0.5 0.5 0.2 0.2\n", classes, YoloFormat::Boxes,
                            QStringLiteral("image.png"), QSize(100, 100))
                         .error.isEmpty());
    ASSERT_FALSE(importYolo("1 nan 0.5 0.2 0.2\n", classes, YoloFormat::Boxes,
                            QStringLiteral("image.png"), QSize(100, 100))
                         .error.isEmpty());
    ASSERT_FALSE(importYolo("1 0.2 0.3 0.4\n", classes, YoloFormat::Segments,
                            QStringLiteral("image.png"), QSize(100, 100))
                         .error.isEmpty());
}

TEST_F(AnnotationExporterTest,
       YoloPoseRoundTripPreservesGroupingAndVisibilityPadding) {
    document.insert(QStringLiteral("shapes"), QJsonArray{poseBox, nose});
    const YoloConversion pose =
            convertYolo(document, {}, YoloFormat::Pose, true, &poseConfig);
    ASSERT_TRUE(poseConfig.error.isEmpty()) << poseConfig.error.toStdString();
    ASSERT_EQ(pose.count, 1);
    ASSERT_EQ(pose.text, "0 0.3 0.4 0.4 0.4 0.2 0.3 2 0 0 0 0 0 0\n");
    const YoloImport importedPose = importYolo(pose.text, {}, YoloFormat::Pose,
                                               QStringLiteral("image.png"),
                                               QSize(100, 100), &poseConfig);
    const QJsonArray importedPoseItems =
            importedPose.document.value(QStringLiteral("shapes")).toArray();
    ASSERT_TRUE(importedPose.error.isEmpty())
            << importedPose.error.toStdString();
    ASSERT_EQ(importedPose.count, 1);
    ASSERT_EQ(importedPoseItems.size(), 2);
    ASSERT_EQ(
            importedPoseItems[0].toObject().value(QStringLiteral("group_id")),
            importedPoseItems[1].toObject().value(QStringLiteral("group_id")));
    ASSERT_EQ(importedPoseItems[1]
                      .toObject()
                      .value(QStringLiteral("label"))
                      .toString(),
              QStringLiteral("nose"));
    const PoseConfig noVisibility = parsePoseConfig(
            "has_visible: false\nclasses:\n  person:\n    - nose\n    - eye\n");
    ASSERT_EQ(convertYoloPose(document, noVisibility).text,
              "0 0.3 0.4 0.4 0.4 0.2 0.3 0 0\n");
}

TEST_F(AnnotationExporterTest,
       PoseRejectsUnmatchedGroupsMalformedYamlAndMissingConfiguration) {
    nose.insert(QStringLiteral("group_id"), 5);
    document.insert(QStringLiteral("shapes"), QJsonArray{poseBox, nose});
    ASSERT_FALSE(convertYoloPose(document, poseConfig).error.isEmpty());
    ASSERT_FALSE(
            parsePoseConfig(
                    "has_visible: maybe\nclasses:\n  person:\n    - nose\n")
                    .error.isEmpty());
    ASSERT_FALSE(parsePoseConfig("- just a list\n").error.isEmpty());
    ASSERT_FALSE(parsePoseConfig("classes:\n\t- x\n").error.isEmpty());
    ASSERT_FALSE(importYolo("0 0.5 0.5 0.2 0.2\n", classes, YoloFormat::Pose,
                            QStringLiteral("image.png"), QSize(100, 100))
                         .error.isEmpty());
}

TEST_F(AnnotationExporterTest,
       PoseYamlAcceptsFlowIndentationAliasesAndDuplicateKeypoints) {
    // The source reads pose configs through yaml.safe_load, so the parser
    // must accept additional valid YAML constructs beyond the canonical
    // schema: any consistent indentation, sequence items at the parent
    // key's indent, flow style, comments, extra root keys, quoted scalars,
    // YAML 1.1 booleans, anchors/aliases and duplicate mapping keys (last
    // value wins) with duplicate keypoint entries preserved.
    nose.insert(QStringLiteral("group_id"), 4);
    nose.remove(QStringLiteral("difficult"));
    document.insert(QStringLiteral("shapes"), QJsonArray{poseBox, nose});
    const YoloConversion pose =
            convertYolo(document, {}, YoloFormat::Pose, true, &poseConfig);
    const PoseConfig flowConfig = parsePoseConfig(
            "# pose template with comments\n"
            "name: demo          # extra root keys are ignored\n"
            "has_visible: yes\n"
            "classes: {person: [nose, eye],\n"
            "          dog: [nose, ear, paw]}\n");
    const PoseConfig indentedConfig = parsePoseConfig(
            "---\n"
            "has_visible: on\n"
            "classes:\n"
            " 'person':\n"
            " - nose\n"
            " - eye\n"
            " dog:\n"
            "   - nose\n"
            "   - ear\n"
            "   - paw\n");
    const PoseConfig aliasConfig = parsePoseConfig(
            "classes:\n"
            "  person: &base\n"
            "    - nose\n"
            "    - eye\n"
            "  dog: *base\n");
    ASSERT_TRUE(flowConfig.error.isEmpty()) << flowConfig.error.toStdString();
    ASSERT_TRUE(indentedConfig.error.isEmpty())
            << indentedConfig.error.toStdString();
    ASSERT_TRUE(aliasConfig.error.isEmpty()) << aliasConfig.error.toStdString();
    const YoloConversion flowPose =
            convertYolo(document, {}, YoloFormat::Pose, true, &flowConfig);
    const YoloConversion indentedPose =
            convertYolo(document, {}, YoloFormat::Pose, true, &indentedConfig);
    ASSERT_EQ(flowPose.text, pose.text);
    ASSERT_EQ(indentedPose.text, pose.text);
    ASSERT_EQ(aliasConfig.classes.size(), 2);
    ASSERT_EQ(aliasConfig.classes[0].keypoints,
              aliasConfig.classes[1].keypoints);
    ASSERT_TRUE(aliasConfig.hasVisible);
    const PoseConfig duplicateConfig = parsePoseConfig(
            "classes:\n"
            "  person:\n"
            "    - nose\n"
            "    - nose\n");
    ASSERT_TRUE(duplicateConfig.error.isEmpty())
            << duplicateConfig.error.toStdString();
    ASSERT_EQ(duplicateConfig.classes.size(), 1);
    ASSERT_EQ(duplicateConfig.classes[0].keypoints.size(), 2);
}

TEST_F(AnnotationExporterTest,
       CocoPoseRoundTripPreservesVisibilityGroupingAndCategories) {
    nose.insert(QStringLiteral("group_id"), 4);
    nose.insert(QStringLiteral("difficult"), true);
    document.insert(QStringLiteral("shapes"), QJsonArray{poseBox, nose});
    const CocoConversion cocoPose =
            convertCocoPose({{QStringLiteral("image.png"), document},
                             {QStringLiteral("next.png"), document}},
                            poseConfig);
    const QJsonArray poseAnnotations =
            cocoPose.document.value(QStringLiteral("annotations")).toArray();
    const QJsonArray poseCategories =
            cocoPose.document.value(QStringLiteral("categories")).toArray();
    ASSERT_TRUE(cocoPose.error.isEmpty()) << cocoPose.error.toStdString();
    ASSERT_EQ(cocoPose.count, 2);
    ASSERT_EQ(poseAnnotations.size(), 2);
    ASSERT_GE(poseCategories.size(), 1);
    ASSERT_EQ(poseAnnotations[0]
                      .toObject()
                      .value(QStringLiteral("bbox"))
                      .toArray(),
              (QJsonArray{10, 20, 40, 40}));
    ASSERT_EQ(poseAnnotations[0]
                      .toObject()
                      .value(QStringLiteral("keypoints"))
                      .toArray(),
              (QJsonArray{20, 30, 1, 0, 0, 0}));
    ASSERT_EQ(poseAnnotations[0]
                      .toObject()
                      .value(QStringLiteral("num_keypoints"))
                      .toInt(),
              1);
    ASSERT_EQ(poseAnnotations[1]
                      .toObject()
                      .value(QStringLiteral("image_id"))
                      .toInt(),
              1);
    ASSERT_EQ(poseCategories[0]
                      .toObject()
                      .value(QStringLiteral("keypoints"))
                      .toArray(),
              (QJsonArray{QStringLiteral("nose"), QStringLiteral("eye")}));
    const CocoImport poseRoundTrip = importCoco(
            QJsonDocument(cocoPose.document).toJson(), CocoFormat::Pose);
    const QJsonArray importedPoseShapes =
            poseRoundTrip.documents.value(QStringLiteral("image.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray();
    ASSERT_TRUE(poseRoundTrip.error.isEmpty())
            << poseRoundTrip.error.toStdString();
    ASSERT_EQ(poseRoundTrip.count, 4);
    ASSERT_EQ(importedPoseShapes.size(), 2);
    ASSERT_EQ(
            importedPoseShapes[0].toObject().value(QStringLiteral("group_id")),
            importedPoseShapes[1].toObject().value(QStringLiteral("group_id")));
    ASSERT_TRUE(importedPoseShapes[1]
                        .toObject()
                        .value(QStringLiteral("difficult"))
                        .toBool());
}

TEST_F(AnnotationExporterTest,
       RgbMaskPriorityDeterminesOverlapAndImportedLabel) {
    const MaskConversion rgbMask = convertMask(maskDocument, rgbMap);
    ASSERT_TRUE(rgbMask.error.isEmpty()) << rgbMask.error.toStdString();
    ASSERT_EQ(rgbMask.image.size(), QSize(16, 16));
    ASSERT_EQ(rgbMask.image.pixelColor(7, 7), QColor(85, 0, 255));
    ASSERT_EQ(rgbMask.image.pixelColor(0, 0), QColor(0, 0, 0));
    const AnnotationImport importedRgbMask = importMask(
            rgbMask.image, rgbMap, QStringLiteral("mask.png"), QSize(16, 16));
    ASSERT_TRUE(importedRgbMask.error.isEmpty())
            << importedRgbMask.error.toStdString();
    ASSERT_EQ(importedRgbMask.count, 1);
    ASSERT_EQ(importedRgbMask.document.value(QStringLiteral("shapes"))
                      .toArray()
                      .size(),
              1);
    ASSERT_EQ(importedRgbMask.document.value(QStringLiteral("shapes"))
                      .toArray()[0]
                      .toObject()
                      .value(QStringLiteral("label"))
                      .toString(),
              QStringLiteral("cat"));
}

#if defined(QANYLABELING_HAS_OPENCV_MASK)
TEST_F(AnnotationExporterTest,
       PolygonMaskMatchesEveryOpenCvFillPolyBoundaryPixel) {
    // The source converter uses cv2.fillPoly. Compare every output pixel,
    // including integer boundary pixels, with that operation.
    const QJsonObject boundaryShape{
            {QStringLiteral("label"), QStringLiteral("cat")},
            {QStringLiteral("shape_type"), QStringLiteral("polygon")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{1, 1}, QJsonArray{7, 3}, QJsonArray{5, 11},
                        QJsonArray{2, 8}}}};
    const QJsonObject boundaryDocument{
            {QStringLiteral("imageWidth"), 16},
            {QStringLiteral("imageHeight"), 16},
            {QStringLiteral("shapes"), QJsonArray{boundaryShape}}};
    const MaskConversion boundaryMask = convertMask(boundaryDocument, rgbMap);
    ASSERT_TRUE(boundaryMask.error.isEmpty())
            << boundaryMask.error.toStdString();
    ASSERT_EQ(boundaryMask.image.size(), QSize(16, 16));
    cv::Mat expected(16, 16, CV_8UC1, cv::Scalar(0));
    const std::vector<cv::Point> contour{{1, 1}, {7, 3}, {5, 11}, {2, 8}};
    cv::fillPoly(expected, std::vector<std::vector<cv::Point>>{contour},
                 cv::Scalar(255));
    for (int y = 0; y < expected.rows; ++y) {
        for (int x = 0; x < expected.cols; ++x) {
            SCOPED_TRACE(::testing::Message() << "pixel " << x << "," << y);
            const bool filled = expected.at<uchar>(y, x) != 0;
            const bool colored =
                    boundaryMask.image.pixelColor(x, y) == QColor(85, 0, 255);
            ASSERT_EQ(filled, colored);
        }
    }
}

#endif

TEST_F(AnnotationExporterTest,
       ExactMaskRoundTripPreservesHolesAndDisconnectedIslands) {
    QImage island(16, 16, QImage::Format_ARGB32);
    island.fill(Qt::transparent);
    island.setPixel(4, 4, qRgba(255, 255, 255, 255));
    island.setPixel(12, 12, qRgba(255, 255, 255, 255));
    QByteArray islandPng;
    QBuffer islandBuffer(&islandPng);
    ASSERT_TRUE(islandBuffer.open(QIODevice::WriteOnly));
    ASSERT_TRUE(island.save(&islandBuffer, "PNG"));
    const QJsonObject exactShape{
            {QStringLiteral("label"), QStringLiteral("cat")},
            {QStringLiteral("shape_type"), QStringLiteral("polygon")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{3, 3}, QJsonArray{13, 3}, QJsonArray{13, 13},
                        QJsonArray{3, 13}}},
            {QStringLiteral("acloudviewer_mask_png"),
             QString::fromLatin1(islandPng.toBase64())}};
    const QJsonObject exactDocument{
            {QStringLiteral("imageWidth"), 16},
            {QStringLiteral("imageHeight"), 16},
            {QStringLiteral("shapes"), QJsonArray{exactShape}}};
    const MaskConversion exactExport = convertMask(exactDocument, rgbMap);
    ASSERT_TRUE(exactExport.error.isEmpty()) << exactExport.error.toStdString();
    ASSERT_EQ(exactExport.image.pixelColor(4, 4), QColor(85, 0, 255));
    ASSERT_EQ(exactExport.image.pixelColor(12, 12), QColor(85, 0, 255));
    ASSERT_EQ(exactExport.image.pixelColor(8, 8), QColor(0, 0, 0));
    const AnnotationImport exactImport =
            importMask(exactExport.image, rgbMap, QStringLiteral("mask.png"),
                       QSize(16, 16));
    const QJsonArray exactShapes =
            exactImport.document.value(QStringLiteral("shapes")).toArray();
    ASSERT_TRUE(exactImport.error.isEmpty()) << exactImport.error.toStdString();
    ASSERT_EQ(exactShapes.size(), 1);
    ASSERT_FALSE(exactShapes.first()
                         .toObject()
                         .value(QStringLiteral("acloudviewer_mask_png"))
                         .toString()
                         .isEmpty());
    ASSERT_EQ(convertMask(exactImport.document, rgbMap).image,
              exactExport.image);
}

TEST_F(AnnotationExporterTest,
       GrayscaleMaskRoundTripPreservesLabelsAndRejectsWrongDimensions) {
    const QJsonObject grayMap{
            {QStringLiteral("type"), QStringLiteral("grayscale")},
            {QStringLiteral("colors"),
             QJsonObject{{QStringLiteral("cat"), 8},
                         {QStringLiteral("dog"), 3}}}};
    const MaskConversion grayMask = convertMask(maskDocument, grayMap);
    ASSERT_TRUE(grayMask.error.isEmpty()) << grayMask.error.toStdString();
    ASSERT_EQ(grayMask.image.format(), QImage::Format_Grayscale8);
    ASSERT_EQ(grayMask.image.pixelColor(7, 7), QColor(3, 3, 3));
    const AnnotationImport importedGrayMask = importMask(
            grayMask.image, grayMap, QStringLiteral("mask.png"), QSize(16, 16));
    ASSERT_TRUE(importedGrayMask.error.isEmpty())
            << importedGrayMask.error.toStdString();
    ASSERT_EQ(importedGrayMask.count, 2);
    ASSERT_FALSE(importMask(grayMask.image, grayMap, QStringLiteral("mask.png"),
                            QSize(10, 10))
                         .error.isEmpty());
}

TEST_F(AnnotationExporterTest, MaskRejectsUnknownPriorityLabels) {
    QJsonObject badMap = rgbMap;
    badMap.insert(QStringLiteral("label_priority"),
                  QJsonObject{{QStringLiteral("unknown"), 1}});
    ASSERT_FALSE(convertMask(maskDocument, badMap).error.isEmpty());
}

TEST_F(AnnotationExporterTest, YoloRejectsMissingImageDimensions) {
    document.remove(QStringLiteral("imageWidth"));
    ASSERT_FALSE(
            convertYolo(document, classes, YoloFormat::Boxes).error.isEmpty());
}

TEST_F(AnnotationExporterTest,
       DotaRoundTripPreservesRotationAndRejectsInvalidDifficulty) {
    QJsonObject inBounds = oriented;
    inBounds.insert(QStringLiteral("points"),
                    QJsonArray{QJsonArray{10, 20}, QJsonArray{50, 20},
                               QJsonArray{50, 60}, QJsonArray{10, 60}});
    document.insert(QStringLiteral("shapes"), QJsonArray{inBounds});
    ASSERT_EQ(convertDota(document).text, "10 20 50 20 50 60 10 60 bus 0\n");
    const AnnotationImport importedDota =
            importDota(convertDota(document).text, QStringLiteral("image.png"),
                       QSize(100, 100));
    ASSERT_TRUE(importedDota.error.isEmpty())
            << importedDota.error.toStdString();
    ASSERT_EQ(importedDota.count, 1);
    ASSERT_EQ(importedDota.document.value(QStringLiteral("shapes"))
                      .toArray()
                      .size(),
              1);
    const QJsonObject dotaShape =
            importedDota.document.value(QStringLiteral("shapes"))
                    .toArray()
                    .at(0)
                    .toObject();
    ASSERT_TRUE(importedDota.error.isEmpty())
            << importedDota.error.toStdString();
    ASSERT_EQ(importedDota.count, 1);
    ASSERT_EQ(dotaShape.value(QStringLiteral("shape_type")).toString(),
              QStringLiteral("rotation"));
    ASSERT_EQ(dotaShape.value(QStringLiteral("direction")).toDouble(), 0.0);
    ASSERT_FALSE(importDota("10 20 50 20 50 60 10 60 bus nope\n",
                            QStringLiteral("image.png"), QSize(100, 100))
                         .error.isEmpty());
}

TEST_F(AnnotationExporterTest,
       CocoRoundTripPreservesGroupedSegmentsAndRejectsMissingImage) {
    QJsonObject grouped = polygon;
    grouped.insert(QStringLiteral("group_id"), 7);
    grouped.insert(QStringLiteral("points"),
                   QJsonArray{QJsonArray{0, 0}, QJsonArray{10, 0},
                              QJsonArray{10, 10}, QJsonArray{0, 10}});
    QJsonObject second = grouped;
    second.insert(QStringLiteral("points"),
                  QJsonArray{QJsonArray{20, 20}, QJsonArray{30, 20},
                             QJsonArray{30, 30}, QJsonArray{20, 30}});
    document.insert(QStringLiteral("shapes"), QJsonArray{box, grouped, second});
    const QVector<CocoImage> images{{QStringLiteral("image.png"), document}};
    const CocoConversion cocoBoxes = convertCoco(images, classes, false);
    const CocoConversion cocoSegments = convertCoco(images, classes, true);
    const QJsonArray boxAnnotations =
            cocoBoxes.document.value(QStringLiteral("annotations")).toArray();
    const QJsonArray segmentAnnotations =
            cocoSegments.document.value(QStringLiteral("annotations"))
                    .toArray();
    ASSERT_TRUE(cocoBoxes.error.isEmpty()) << cocoBoxes.error.toStdString();
    ASSERT_TRUE(cocoSegments.error.isEmpty())
            << cocoSegments.error.toStdString();
    ASSERT_EQ(cocoBoxes.count, 1);
    ASSERT_EQ(cocoSegments.count, 1);
    ASSERT_EQ(boxAnnotations.size(), 1);
    ASSERT_EQ(segmentAnnotations.size(), 1);
    ASSERT_EQ(boxAnnotations[0]
                      .toObject()
                      .value(QStringLiteral("category_id"))
                      .toInt(),
              2);
    ASSERT_EQ(segmentAnnotations[0]
                      .toObject()
                      .value(QStringLiteral("category_id"))
                      .toInt(),
              1);
    ASSERT_EQ(segmentAnnotations[0]
                      .toObject()
                      .value(QStringLiteral("segmentation"))
                      .toArray()
                      .size(),
              2);
    ASSERT_EQ(cocoSegments.document.value(QStringLiteral("categories"))
                      .toArray()
                      .size(),
              3);
    ASSERT_EQ(cocoSegments.document.value(QStringLiteral("type")).toString(),
              QStringLiteral("instances"));
    const CocoImport importedCocoBox = importCoco(
            QJsonDocument(cocoBoxes.document).toJson(), CocoFormat::Boxes);
    const CocoImport importedCocoSeg =
            importCoco(QJsonDocument(cocoSegments.document).toJson(),
                       CocoFormat::Segments);
    const QJsonArray importedSegments =
            importedCocoSeg.documents.value(QStringLiteral("image.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray();
    ASSERT_TRUE(importedCocoBox.error.isEmpty())
            << importedCocoBox.error.toStdString();
    ASSERT_EQ(importedCocoBox.count, 1);
    ASSERT_TRUE(importedCocoSeg.error.isEmpty())
            << importedCocoSeg.error.toStdString();
    ASSERT_EQ(importedCocoSeg.count, 2);
    ASSERT_EQ(importedSegments.size(), 2);
    ASSERT_EQ(importedSegments[0].toObject().value(QStringLiteral("group_id")),
              importedSegments[1].toObject().value(QStringLiteral("group_id")));
    ASSERT_FALSE(
            importCoco("{\"categories\":[],\"images\":[],\"annotations\":[{}]}",
                       CocoFormat::Boxes)
                    .error.isEmpty());
}

class CocoRleRoundTripTest : public AnnotationExporterTest,
                             public ::testing::WithParamInterface<bool> {};

TEST_P(CocoRleRoundTripTest, PreservesEveryPixelThroughImportExportAndImport) {
    const int rleWidth = 8;
    const int rleHeight = 8;
    const QSet<int> foreground{
            2 * rleHeight + 2, 2 * rleHeight + 3, 2 * rleHeight + 4,
            3 * rleHeight + 2, 3 * rleHeight + 4, 4 * rleHeight + 2,
            4 * rleHeight + 3, 4 * rleHeight + 4, 6 * rleHeight + 6};
    QVector<qint64> rleRuns;
    bool previous = false;
    qint64 runLength = 0;
    for (int index = 0; index < rleWidth * rleHeight; ++index) {
        const bool current = foreground.contains(index);
        if (current != previous) {
            rleRuns.append(runLength);
            runLength = 0;
            previous = current;
        }
        ++runLength;
    }
    rleRuns.append(runLength);
    auto compressRle = [](const QVector<qint64>& runs) {
        QByteArray encoded;
        for (int i = 0; i < runs.size(); ++i) {
            qint64 value = runs[i] - (i > 2 ? runs[i - 2] : 0);
            bool more = true;
            while (more) {
                int symbol = int(value & 0x1f);
                value >>= 5;
                more = (symbol & 0x10) ? value != -1 : value != 0;
                if (more) symbol |= 0x20;
                encoded.append(char(symbol + 48));
            }
        }
        return encoded;
    };
    const QJsonObject rleRoot{
            {QStringLiteral("categories"),
             QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
                                    {QStringLiteral("name"), "ring"}}}},
            {QStringLiteral("images"),
             QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
                                    {QStringLiteral("file_name"), "rle.png"},
                                    {QStringLiteral("width"), rleWidth},
                                    {QStringLiteral("height"), rleHeight}}}},
            {QStringLiteral("annotations"),
             QJsonArray{QJsonObject{
                     {QStringLiteral("image_id"), 1},
                     {QStringLiteral("category_id"), 1},
                     {QStringLiteral("segmentation"), QJsonValue()}}}}};
    const bool compressed = GetParam();
    QJsonObject root = rleRoot;
    QJsonArray annotations =
            root.value(QStringLiteral("annotations")).toArray();
    QJsonObject annotation = annotations.first().toObject();
    QJsonValue counts;
    if (compressed) {
        counts = QString::fromLatin1(compressRle(rleRuns));
    } else {
        QJsonArray values;
        for (const qint64 run : rleRuns) values.append(run);
        counts = values;
    }
    annotation.insert(QStringLiteral("segmentation"),
                      QJsonObject{{QStringLiteral("size"),
                                   QJsonArray{rleHeight, rleWidth}},
                                  {QStringLiteral("counts"), counts}});
    annotations[0] = annotation;
    root.insert(QStringLiteral("annotations"), annotations);
    const CocoImport importedRle =
            importCoco(QJsonDocument(root).toJson(), CocoFormat::Segments);
    const QJsonArray importedShapes =
            importedRle.documents.value(QStringLiteral("rle.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray();
    ASSERT_TRUE(importedRle.error.isEmpty()) << importedRle.error.toStdString();
    ASSERT_EQ(importedRle.count, 1);
    ASSERT_EQ(importedShapes.size(), 1);
    const QJsonObject shape = importedShapes.at(0).toObject();
    const QImage restored = QImage::fromData(
            QByteArray::fromBase64(
                    shape.value(QStringLiteral("acloudviewer_mask_png"))
                            .toString()
                            .toLatin1()),
            "PNG");
    ASSERT_FALSE(restored.isNull());
    ASSERT_NE(restored.pixelColor(2, 2).alpha(), 0);
    ASSERT_EQ(restored.pixelColor(3, 3).alpha(), 0);
    ASSERT_NE(restored.pixelColor(6, 6).alpha(), 0);
    const CocoConversion exportedRle = convertCoco(
            {{QStringLiteral("rle.png"),
              importedRle.documents.value(QStringLiteral("rle.png"))}},
            {QStringLiteral("ring")}, true);
    const QJsonArray exportedAnnotations =
            exportedRle.document.value(QStringLiteral("annotations")).toArray();
    ASSERT_TRUE(exportedRle.error.isEmpty()) << exportedRle.error.toStdString();
    ASSERT_EQ(exportedRle.count, 1);
    ASSERT_EQ(exportedAnnotations.size(), 1);
    ASSERT_TRUE(exportedAnnotations.first()
                        .toObject()
                        .value(QStringLiteral("segmentation"))
                        .isObject());
    const CocoImport roundTrip = importCoco(
            QJsonDocument(exportedRle.document).toJson(), CocoFormat::Segments);
    const QJsonArray roundTripShapes =
            roundTrip.documents.value(QStringLiteral("rle.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray();
    ASSERT_TRUE(roundTrip.error.isEmpty()) << roundTrip.error.toStdString();
    ASSERT_EQ(roundTripShapes.size(), 1);
    const QImage roundTripMask = QImage::fromData(
            QByteArray::fromBase64(
                    roundTripShapes.first()
                            .toObject()
                            .value(QStringLiteral("acloudviewer_mask_png"))
                            .toString()
                            .toLatin1()),
            "PNG");
    ASSERT_EQ(roundTripMask.size(), QSize(rleWidth, rleHeight));
    for (int x = 0; x < rleWidth; ++x) {
        for (int y = 0; y < rleHeight; ++y) {
            SCOPED_TRACE(::testing::Message() << "pixel " << x << "," << y);
            EXPECT_EQ(restored.pixelColor(x, y).alpha() > 0,
                      roundTripMask.pixelColor(x, y).alpha() > 0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Encodings,
                         CocoRleRoundTripTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "Compressed" : "Uncompressed";
                         });

TEST_F(AnnotationExporterTest,
       MotRoundTripPreservesTracksEmptyFramesAndRejectsUnknownClasses) {
    QJsonObject tracked = box;
    tracked.insert(QStringLiteral("group_id"), 9);
    const QVector<MotImage> motImages{
            {QStringLiteral("frame_000001.png"),
             QSize(100, 100),
             {{QStringLiteral("shapes"), QJsonArray{tracked}}}},
            {QStringLiteral("frame_000002.png"),
             QSize(100, 100),
             {{QStringLiteral("shapes"), QJsonArray()}}}};
    const MotConversion mot =
            convertMot(motImages, classes, QStringLiteral("mot"));
    const MotImport importedMot =
            importMot(mot.groundTruth, classes, motImages);
    ASSERT_TRUE(importedMot.error.isEmpty()) << importedMot.error.toStdString();
    ASSERT_EQ(importedMot.count, 1);
    ASSERT_EQ(importedMot.documents.value(QStringLiteral("frame_000001.png"))
                      .value(QStringLiteral("shapes"))
                      .toArray()
                      .size(),
              1);
    const QJsonObject trackedShape =
            importedMot.documents.value(QStringLiteral("frame_000001.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray()[0]
                    .toObject();
    ASSERT_TRUE(mot.error.isEmpty()) << mot.error.toStdString();
    ASSERT_EQ(mot.count, 1);
    ASSERT_EQ(mot.groundTruth, "1,9,10,20,40,40,1,1,1\n");
    ASSERT_EQ(mot.detections, "1,-1,10,20,40,40,1,-1,-1,-1\n");
    ASSERT_TRUE(mot.sequence.contains("seqlength = 2"));
    ASSERT_TRUE(importedMot.error.isEmpty()) << importedMot.error.toStdString();
    ASSERT_EQ(importedMot.count, 1);
    ASSERT_EQ(trackedShape.value(QStringLiteral("group_id")).toInt(), 9);
    ASSERT_FALSE(importMot("1,9,10,20,40,40,1,99,1\n", classes, motImages)
                         .error.isEmpty());
}

TEST_F(AnnotationExporterTest,
       MotsExportsClippedTrackedPolygonsAndRejectsInvalidGeometry) {
    QJsonObject trackedPolygon = polygon;
    trackedPolygon.insert(QStringLiteral("group_id"), 12);
    const QVector<MotImage> motsImages{
            {QStringLiteral("frame_000001.png"),
             QSize(100, 100),
             {{QStringLiteral("shapes"), QJsonArray{trackedPolygon}}}},
            {QStringLiteral("frame_000002.png"),
             QSize(100, 100),
             {{QStringLiteral("shapes"), QJsonArray()}}}};
    const MotConversion mots =
            convertMots(motsImages, classes, QStringLiteral("mots"));
    ASSERT_TRUE(mots.error.isEmpty()) << mots.error.toStdString();
    ASSERT_EQ(mots.count, 1);
    ASSERT_EQ(mots.groundTruth, "1 12 0 100 100 [[0, 0], [50, 0], [50, 99]]\n");
    ASSERT_TRUE(mots.sequence.contains("name = MOTS"));
    ASSERT_TRUE(mots.sequence.contains("seqlength = 2"));
    ASSERT_TRUE(mots.detections.isEmpty());
    QJsonObject invalidPolygon = trackedPolygon;
    invalidPolygon.insert(QStringLiteral("points"),
                          QJsonArray{QJsonArray{1, 2}});
    const QVector<MotImage> invalidMots{
            {QStringLiteral("frame_000001.png"),
             QSize(100, 100),
             {{QStringLiteral("shapes"), QJsonArray{invalidPolygon}}}}};
    ASSERT_FALSE(convertMots(invalidMots, classes, QStringLiteral("mots"))
                         .error.isEmpty());
}

TEST_F(AnnotationExporterTest,
       VlmRoundTripPreservesEmptyImagesAndRejectsMalformedOrDuplicateRecords) {
    const QVector<CocoImage> odvgImages{
            {QStringLiteral("image.png"),
             {{QStringLiteral("imageWidth"), 100},
              {QStringLiteral("imageHeight"), 100},
              {QStringLiteral("shapes"), QJsonArray{box}}}},
            {QStringLiteral("next.png"),
             {{QStringLiteral("imageWidth"), 100},
              {QStringLiteral("imageHeight"), 100},
              {QStringLiteral("shapes"), QJsonArray()}}}};
    const VlmConversion vlm =
            convertVlm(odvgImages, classes, QStringLiteral("images/"));
    const VlmImport importedVlm = importVlm(vlm.annotations);
    const QJsonArray vlmShapes =
            importedVlm.documents.value(QStringLiteral("image.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray();
    ASSERT_TRUE(vlm.error.isEmpty()) << vlm.error.toStdString();
    ASSERT_EQ(vlm.count, 1);
    ASSERT_EQ(importedVlm.count, 1);
    ASSERT_EQ(importedVlm.documents.size(), 2);
    ASSERT_EQ(vlmShapes.size(), 1);
    ASSERT_EQ(vlmShapes[0].toObject().value(QStringLiteral("label")).toString(),
              QStringLiteral("bus"));
    ASSERT_TRUE(vlm.annotations.contains("images/image.png"));
    ASSERT_EQ(importedVlm.documents.value(QStringLiteral("next.png"))
                      .value(QStringLiteral("shapes"))
                      .toArray()
                      .size(),
              0);
    ASSERT_FALSE(importVlm("{bad json}\n").error.isEmpty());
    ASSERT_FALSE(importVlm(vlm.annotations + vlm.annotations).error.isEmpty());
}

TEST_F(AnnotationExporterTest, VlmRepairsJsonLikeAnswerSyntax) {
    const QByteArray repairedVlm(
            "{\"image\":\"image.png\",\"conversations\":[{},"
            "{\"value\":\"<answer>[{bbox_2d: [1, 2, 5, 6], label: "
            "'bus',},]</answer>\"}]}\n");
    const VlmImport repairedImport = importVlm(repairedVlm);
    ASSERT_TRUE(repairedImport.error.isEmpty())
            << repairedImport.error.toStdString();
    ASSERT_EQ(repairedImport.documents.value(QStringLiteral("image.png"))
                      .value(QStringLiteral("shapes"))
                      .toArray()
                      .size(),
              1);
}

TEST_F(AnnotationExporterTest,
       OdvgRoundTripPreservesEmptyImagesAndRejectsDuplicateOrMissingData) {
    const QVector<CocoImage> odvgImages{
            {QStringLiteral("image.png"),
             {{QStringLiteral("imageWidth"), 100},
              {QStringLiteral("imageHeight"), 100},
              {QStringLiteral("shapes"), QJsonArray{box}}}},
            {QStringLiteral("next.png"),
             {{QStringLiteral("imageWidth"), 100},
              {QStringLiteral("imageHeight"), 100},
              {QStringLiteral("shapes"), QJsonArray()}}}};
    const OdvgConversion odvg = convertOdvg(odvgImages, classes);
    const OdvgImport importedOdvg = importOdvg(odvg.annotations);
    const QJsonArray odvgShapes =
            importedOdvg.documents.value(QStringLiteral("image.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray();
    ASSERT_TRUE(odvg.error.isEmpty()) << odvg.error.toStdString();
    ASSERT_EQ(odvg.count, 1);
    ASSERT_TRUE(QJsonDocument::fromJson(odvg.labelMap).isObject());
    ASSERT_TRUE(importedOdvg.error.isEmpty())
            << importedOdvg.error.toStdString();
    ASSERT_EQ(importedOdvg.count, 1);
    ASSERT_EQ(importedOdvg.documents.size(), 2);
    ASSERT_EQ(odvgShapes.size(), 1);
    ASSERT_EQ(
            odvgShapes[0].toObject().value(QStringLiteral("label")).toString(),
            QStringLiteral("bus"));
    ASSERT_FALSE(importOdvg(odvg.annotations +
                            odvg.annotations.split('\n').first() + '\n')
                         .error.isEmpty());
    ASSERT_FALSE(importOdvg("{\"filename\":\"x.png\"}\n").error.isEmpty());
}

}  // namespace
