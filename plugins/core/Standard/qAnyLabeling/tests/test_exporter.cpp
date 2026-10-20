// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QBuffer>
#include <QCoreApplication>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSet>
#include <QTemporaryDir>
#include <QVector>
#include <cstdio>

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

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    const QJsonObject ocrShape{
            {QStringLiteral("label"), QStringLiteral("text")},
            {QStringLiteral("description"), QStringLiteral("hello")},
            {QStringLiteral("shape_type"), QStringLiteral("quadrilateral")},
            {QStringLiteral("group_id"), 3},
            {QStringLiteral("difficult"), true},
            {QStringLiteral("kie_linking"), QJsonArray{QJsonArray{3, 4}}},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{1, 2}, QJsonArray{30, 2}, QJsonArray{30, 10},
                        QJsonArray{1, 10}}}};
    const QJsonObject ocrDocument{
            {QStringLiteral("shapes"), QJsonArray{ocrShape}}};
    const PpocrConversion rec = convertPpocr(
            {{QStringLiteral("ocr.png"), ocrDocument}}, PpocrMode::Rec);
    const PpocrConversion kie = convertPpocr(
            {{QStringLiteral("ocr.png"), ocrDocument}}, PpocrMode::Kie);
    const PpocrImport recImported = importPpocr(rec.text, PpocrMode::Rec);
    const PpocrImport kieImported = importPpocr(kie.text, PpocrMode::Kie);
    if (!rec.error.isEmpty() || !kie.error.isEmpty() ||
        !recImported.error.isEmpty() || !kieImported.error.isEmpty() ||
        rec.count != 1 || kie.count != 1 || recImported.count != 1 ||
        kieImported.count != 1 ||
        kieImported.documents.value(QStringLiteral("ocr.png"))
                        .value(QStringLiteral("shapes"))
                        .toArray()
                        .first()
                        .toObject()
                        .value(QStringLiteral("kie_linking"))
                        .toArray()
                        .size() != 1 ||
        importPpocr("bad\n", PpocrMode::Rec).error.isEmpty()) {
        std::fprintf(stderr, "PPOCR conversion failed\n");
        return 1;
    }
    QTemporaryDir ppocrDirectory;
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
    if (!linkedKie.error.isEmpty() || !linkedImport.error.isEmpty() ||
        linkedKie.count != 4 || linkedImport.count != 4 ||
        exportedEntities.size() != 4 ||
        exportedEntities[0].toObject()["points"].toArray()[0].toArray() !=
                QJsonArray{-1, 2} ||
        exportedEntities[0].toObject()["transcription"] !=
                QStringLiteral("姓名？") ||
        exportedEntities[1].toObject()["linking"].toArray() !=
                QJsonArray{QJsonArray{3, 4}} ||
        !exportedEntities[2].toObject()["id"].isNull() ||
        exportedEntities[3].toObject()["id"].toInt(-1) != 0 ||
        reexport.text != linkedKie.text) {
        std::fprintf(stderr, "PPOCR KIE linked entity roundtrip failed\n");
        return 1;
    }
    QImage ppocrImage(40, 20, QImage::Format_RGB32);
    ppocrImage.fill(Qt::white);
    const QString ppocrImagePath =
            ppocrDirectory.filePath(QStringLiteral("ocr.png"));
    const QString ppocrOutput =
            ppocrDirectory.filePath(QStringLiteral("dataset"));
    if (!ppocrDirectory.isValid() || !ppocrImage.save(ppocrImagePath)) return 1;
    const PpocrRecExport ppocrExport =
            exportPpocrRecDataset({{ppocrImagePath, ocrDocument}}, ppocrOutput);
    QFile ppocrLabels(QDir(ppocrOutput).filePath(QStringLiteral("Label.txt")));
    QFile ppocrGt(QDir(ppocrOutput).filePath(QStringLiteral("rec_gt.txt")));
    const QString cropPath =
            QDir(ppocrOutput)
                    .filePath(QStringLiteral("crop_img/ocr_crop_0.jpg"));
    if (!ppocrExport.error.isEmpty() || ppocrExport.annotationCount != 1 ||
        ppocrExport.cropCount != 1 || !ppocrLabels.open(QIODevice::ReadOnly) ||
        !ppocrGt.open(QIODevice::ReadOnly) || QImage(cropPath).isNull() ||
        !ppocrLabels.readAll().contains("ocr.png\t") ||
        !ppocrGt.readAll().contains("crop_img/ocr_crop_0.jpg\thello\n")) {
        std::fprintf(stderr, "PPOCR Rec dataset export failed\n");
        return 1;
    }
    const PpocrRecExport ppocrEmpty = exportPpocrRecDataset({}, ppocrOutput);
    if (!ppocrEmpty.error.isEmpty() || QFile::exists(cropPath) ||
        QFileInfo(ppocrLabels.fileName()).size() != 0 ||
        QFileInfo(ppocrGt.fileName()).size() != 0) {
        std::fprintf(stderr, "PPOCR empty export failed\n");
        return 1;
    }
    QImage orientedImage(40, 20, QImage::Format_RGB32);
    orientedImage.fill(Qt::red);
    for (int y = 0; y < orientedImage.height(); ++y)
        for (int x = 20; x < orientedImage.width(); ++x)
            orientedImage.setPixelColor(x, y, Qt::blue);
    if (!orientedImage.save(ppocrImagePath)) return 1;
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
    if (!orientedExport.error.isEmpty() || orientedExport.cropCount != 1 ||
        orientedCrop.isNull() || orientedCrop.width() < orientedCrop.height() ||
        orientedCrop.pixelColor(2, orientedCrop.height() / 2).red() < 200 ||
        orientedCrop.pixelColor(orientedCrop.width() - 3,
                                orientedCrop.height() / 2)
                        .blue() < 200 ||
        !orientedGt.open(QIODevice::ReadOnly) ||
        !orientedGt.readAll().contains(QStringLiteral("中文").toUtf8())) {
        std::fprintf(stderr, "PPOCR oriented crop failed\n");
        return 1;
    }
    const QJsonObject box{
            {QStringLiteral("label"), QStringLiteral("bus")},
            {QStringLiteral("shape_type"), QStringLiteral("rectangle")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{10, 20}, QJsonArray{50, 60}}}};
    const QJsonObject polygon{
            {QStringLiteral("label"), QStringLiteral("person")},
            {QStringLiteral("shape_type"), QStringLiteral("polygon")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{-10, 0}, QJsonArray{50, 0},
                        QJsonArray{50, 100}}}};
    const QJsonObject oriented{
            {QStringLiteral("label"), QStringLiteral("bus")},
            {QStringLiteral("shape_type"), QStringLiteral("rotation")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{-10, 20}, QJsonArray{50, 20},
                        QJsonArray{50, 60}, QJsonArray{-10, 60}}}};
    QJsonObject document{
            {QStringLiteral("imageWidth"), 100},
            {QStringLiteral("imageHeight"), 100},
            {QStringLiteral("shapes"), QJsonArray{box, polygon, oriented}}};
    const QStringList classes{QStringLiteral("person"), QStringLiteral("bus")};
    const YoloConversion boxes =
            convertYolo(document, classes, YoloFormat::Boxes);
    const YoloConversion segments =
            convertYolo(document, classes, YoloFormat::Segments);
    const YoloConversion obb =
            convertYolo(document, classes, YoloFormat::OrientedBoxes);
    const YoloConversion skipped =
            convertYolo(document, classes, YoloFormat::OrientedBoxes, false);
    const YoloConversion dotaSkipped = convertDota(document);
    const YoloConversion vocBoxes =
            convertVoc(document, QStringLiteral("image.png"),
                       QStringLiteral("labels"), false);
    const YoloConversion vocPolygons =
            convertVoc(document, QStringLiteral("image.png"),
                       QStringLiteral("labels"), true);
    if (boxes.count != 1 ||
        boxes.text != "1 0.300000 0.400000 0.400000 0.400000\n" ||
        segments.count != 1 ||
        segments.text !=
                "0 0.000000 0.000000 0.500000 0.000000 0.500000 1.000000\n" ||
        obb.count != 1 ||
        obb.text !=
                "1 -0.100000 0.200000 0.500000 0.200000 0.500000 0.600000 "
                "-0.100000 0.600000\n" ||
        skipped.count != 0 || !skipped.text.isEmpty() ||
        dotaSkipped.count != 0 || vocBoxes.count != 1 ||
        vocPolygons.count != 2 || !vocPolygons.text.contains("<x1>0</x1>") ||
        !vocPolygons.text.contains("<filename>image.png</filename>")) {
        std::fprintf(stderr, "YOLO format conversion failed\n");
        return 1;
    }
    const AnnotationImport importedVocBox =
            importVoc(vocBoxes.text, QStringLiteral("image.png"), false);
    const AnnotationImport importedVocPolygons =
            importVoc(vocPolygons.text, QStringLiteral("image.png"), true);
    const QJsonArray vocShapes =
            importedVocPolygons.document.value(QStringLiteral("shapes"))
                    .toArray();
    if (!importedVocBox.error.isEmpty() || importedVocBox.count != 1 ||
        !importedVocPolygons.error.isEmpty() ||
        importedVocPolygons.count != 2 ||
        vocShapes[0].toObject()
                        .value(QStringLiteral("shape_type"))
                        .toString() != QStringLiteral("rectangle") ||
        vocShapes[1].toObject()
                        .value(QStringLiteral("shape_type"))
                        .toString() != QStringLiteral("polygon") ||
        importVoc("<annotation><object></annotation>",
                  QStringLiteral("image.png"), false)
                .error.isEmpty()) {
        std::fprintf(stderr, "VOC import round-trip failed\n");
        return 1;
    }
    for (const auto format :
         {YoloFormat::Boxes, YoloFormat::Segments, YoloFormat::OrientedBoxes}) {
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
        if (!imported.error.isEmpty() || imported.count != 1 ||
            shapes.size() != 1 ||
            shapes[0].toObject()
                            .value(QStringLiteral("shape_type"))
                            .toString() != type ||
            shapes[0].toObject().value(QStringLiteral("group_id")).type() !=
                    QJsonValue::Null ||
            imported.document.value(QStringLiteral("imagePath")).toString() !=
                    QStringLiteral("image.png")) {
            std::fprintf(stderr, "YOLO import round-trip failed\n");
            return 1;
        }
    }
    if (importYolo("9 0.5 0.5 0.2 0.2\n", classes, YoloFormat::Boxes,
                   QStringLiteral("image.png"), QSize(100, 100))
                .error.isEmpty() ||
        importYolo("1 nan 0.5 0.2 0.2\n", classes, YoloFormat::Boxes,
                   QStringLiteral("image.png"), QSize(100, 100))
                .error.isEmpty() ||
        importYolo("1 0.2 0.3 0.4\n", classes, YoloFormat::Segments,
                   QStringLiteral("image.png"), QSize(100, 100))
                .error.isEmpty()) {
        std::fprintf(stderr, "Invalid YOLO labels were accepted\n");
        return 1;
    }
    const PoseConfig poseConfig = parsePoseConfig(
            "has_visible: true\nclasses:\n  person:\n    - nose\n    - eye\n"
            "  dog:\n    - nose\n    - ear\n    - paw\n");
    QJsonObject poseBox = box;
    poseBox.insert(QStringLiteral("label"), QStringLiteral("person"));
    poseBox.insert(QStringLiteral("group_id"), 4);
    QJsonObject nose{
            {QStringLiteral("label"), QStringLiteral("nose")},
            {QStringLiteral("shape_type"), QStringLiteral("point")},
            {QStringLiteral("group_id"), 4},
            {QStringLiteral("points"), QJsonArray{QJsonArray{20.9, 30.2}}}};
    document.insert(QStringLiteral("shapes"), QJsonArray{poseBox, nose});
    const YoloConversion pose =
            convertYolo(document, {}, YoloFormat::Pose, true, &poseConfig);
    if (!poseConfig.error.isEmpty() || pose.count != 1 ||
        pose.text != "0 0.3 0.4 0.4 0.4 0.2 0.3 2 0 0 0 0 0 0\n") {
        std::fprintf(stderr, "YOLO pose conversion failed: %s\n",
                     pose.text.constData());
        return 1;
    }
    const YoloImport importedPose = importYolo(pose.text, {}, YoloFormat::Pose,
                                               QStringLiteral("image.png"),
                                               QSize(100, 100), &poseConfig);
    const QJsonArray importedPoseItems =
            importedPose.document.value(QStringLiteral("shapes")).toArray();
    if (!importedPose.error.isEmpty() || importedPose.count != 1 ||
        importedPoseItems.size() != 2 ||
        importedPoseItems[0].toObject().value(QStringLiteral("group_id")) !=
                importedPoseItems[1].toObject().value(
                        QStringLiteral("group_id")) ||
        importedPoseItems[1]
                        .toObject()
                        .value(QStringLiteral("label"))
                        .toString() != QStringLiteral("nose")) {
        std::fprintf(stderr, "YOLO pose import round-trip failed\n");
        return 1;
    }
    const PoseConfig noVisibility = parsePoseConfig(
            "has_visible: false\nclasses:\n  person:\n    - nose\n    - eye\n");
    if (convertYoloPose(document, noVisibility).text !=
        "0 0.3 0.4 0.4 0.4 0.2 0.3 0 0\n") {
        std::fprintf(stderr, "YOLO pose visibility-free conversion failed\n");
        return 1;
    }
    nose.insert(QStringLiteral("group_id"), 5);
    document.insert(QStringLiteral("shapes"), QJsonArray{poseBox, nose});
    if (convertYoloPose(document, poseConfig).error.isEmpty() ||
        parsePoseConfig("has_visible: maybe\nclasses:\n  person:\n    - nose\n")
                .error.isEmpty() ||
        parsePoseConfig("- just a list\n").error.isEmpty() ||
        parsePoseConfig("classes:\n\t- x\n").error.isEmpty() ||
        importYolo("0 0.5 0.5 0.2 0.2\n", classes, YoloFormat::Pose,
                   QStringLiteral("image.png"), QSize(100, 100))
                .error.isEmpty()) {
        std::fprintf(stderr,
                     "Invalid pose configuration or group was accepted\n");
        return 1;
    }
    // The source reads pose configs through yaml.safe_load, so the parser
    // must accept additional valid YAML constructs beyond the canonical
    // schema: any consistent indentation, sequence items at the parent
    // key's indent, flow style, comments, extra root keys, quoted scalars,
    // YAML 1.1 booleans, anchors/aliases and duplicate mapping keys (last
    // value wins) with duplicate keypoint entries preserved.
    nose.insert(QStringLiteral("group_id"), 4);
    nose.remove(QStringLiteral("difficult"));
    document.insert(QStringLiteral("shapes"), QJsonArray{poseBox, nose});
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
    if (!flowConfig.error.isEmpty() || !indentedConfig.error.isEmpty() ||
        !aliasConfig.error.isEmpty()) {
        std::fprintf(stderr,
                     "Alternate YAML pose syntax was rejected: %s/%s/%s\n",
                     flowConfig.error.toLocal8Bit().constData(),
                     indentedConfig.error.toLocal8Bit().constData(),
                     aliasConfig.error.toLocal8Bit().constData());
        return 1;
    }
    const YoloConversion flowPose =
            convertYolo(document, {}, YoloFormat::Pose, true, &flowConfig);
    const YoloConversion indentedPose =
            convertYolo(document, {}, YoloFormat::Pose, true, &indentedConfig);
    if (flowPose.text != pose.text || indentedPose.text != pose.text) {
        std::fprintf(
                stderr,
                "Alternate YAML pose syntax changed conversion output\n"
                "canonical=[%s]\nflow=[%s] err=[%s]\nindented=[%s] err=[%s]\n",
                pose.text.constData(), flowPose.text.constData(),
                flowPose.error.toLocal8Bit().constData(),
                indentedPose.text.constData(),
                indentedPose.error.toLocal8Bit().constData());
        return 1;
    }
    if (aliasConfig.classes.size() != 2 ||
        aliasConfig.classes[0].keypoints != aliasConfig.classes[1].keypoints ||
        !aliasConfig.hasVisible) {
        std::fprintf(stderr, "YAML anchors were not resolved\n");
        return 1;
    }
    const PoseConfig duplicateConfig = parsePoseConfig(
            "classes:\n"
            "  person:\n"
            "    - nose\n"
            "    - nose\n");
    if (!duplicateConfig.error.isEmpty() ||
        duplicateConfig.classes.size() != 1 ||
        duplicateConfig.classes[0].keypoints.size() != 2) {
        std::fprintf(stderr, "Duplicate keypoint semantics diverged\n");
        return 1;
    }
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
    if (!cocoPose.error.isEmpty() || cocoPose.count != 2 ||
        poseAnnotations[0].toObject().value(QStringLiteral("bbox")).toArray() !=
                QJsonArray{10, 20, 40, 40} ||
        poseAnnotations[0]
                        .toObject()
                        .value(QStringLiteral("keypoints"))
                        .toArray() != QJsonArray{20, 30, 1, 0, 0, 0} ||
        poseAnnotations[0]
                        .toObject()
                        .value(QStringLiteral("num_keypoints"))
                        .toInt() != 1 ||
        poseAnnotations[1]
                        .toObject()
                        .value(QStringLiteral("image_id"))
                        .toInt() != 1 ||
        poseCategories[0]
                        .toObject()
                        .value(QStringLiteral("keypoints"))
                        .toArray() !=
                QJsonArray{QStringLiteral("nose"), QStringLiteral("eye")}) {
        std::fprintf(stderr, "COCO keypoints conversion failed\n");
        return 1;
    }
    const CocoImport poseRoundTrip = importCoco(
            QJsonDocument(cocoPose.document).toJson(), CocoFormat::Pose);
    const QJsonArray importedPoseShapes =
            poseRoundTrip.documents.value(QStringLiteral("image.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray();
    if (!poseRoundTrip.error.isEmpty() || poseRoundTrip.count != 4 ||
        importedPoseShapes.size() != 2 ||
        importedPoseShapes[0].toObject().value(QStringLiteral("group_id")) !=
                importedPoseShapes[1].toObject().value(
                        QStringLiteral("group_id")) ||
        !importedPoseShapes[1]
                 .toObject()
                 .value(QStringLiteral("difficult"))
                 .toBool()) {
        std::fprintf(stderr, "COCO pose import round-trip failed\n");
        return 1;
    }
    const QJsonObject cat{
            {QStringLiteral("label"), QStringLiteral("cat")},
            {QStringLiteral("shape_type"), QStringLiteral("polygon")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{1, 1}, QJsonArray{14, 1}, QJsonArray{14, 14},
                        QJsonArray{1, 14}}}};
    const QJsonObject dog{
            {QStringLiteral("label"), QStringLiteral("dog")},
            {QStringLiteral("shape_type"), QStringLiteral("polygon")},
            {QStringLiteral("points"),
             QJsonArray{QJsonArray{5, 5}, QJsonArray{10, 5}, QJsonArray{10, 10},
                        QJsonArray{5, 10}}}};
    const QJsonObject maskDocument{
            {QStringLiteral("imageWidth"), 16},
            {QStringLiteral("imageHeight"), 16},
            {QStringLiteral("shapes"), QJsonArray{cat, dog}}};
    const QJsonObject rgbMap{
            {QStringLiteral("type"), QStringLiteral("rgb")},
            {QStringLiteral("colors"),
             QJsonObject{{QStringLiteral("cat"), QJsonArray{85, 0, 255}},
                         {QStringLiteral("dog"), QJsonArray{213, 42, 54}}}},
            {QStringLiteral("label_priority"),
             QJsonObject{{QStringLiteral("cat"), 2},
                         {QStringLiteral("dog"), 1}}}};
    const MaskConversion rgbMask = convertMask(maskDocument, rgbMap);
    if (!rgbMask.error.isEmpty() || rgbMask.image.size() != QSize(16, 16) ||
        rgbMask.image.pixelColor(7, 7) != QColor(85, 0, 255) ||
        rgbMask.image.pixelColor(0, 0) != QColor(0, 0, 0)) {
        std::fprintf(stderr, "RGB mask priority or color failed\n");
        return 1;
    }
#if defined(QANYLABELING_HAS_OPENCV_MASK)
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
    cv::Mat expected(16, 16, CV_8UC1, cv::Scalar(0));
    const std::vector<cv::Point> contour{{1, 1}, {7, 3}, {5, 11}, {2, 8}};
    cv::fillPoly(expected, std::vector<std::vector<cv::Point>>{contour},
                 cv::Scalar(255));
    for (int y = 0; y < expected.rows; ++y) {
        for (int x = 0; x < expected.cols; ++x) {
            const bool filled = expected.at<uchar>(y, x) != 0;
            const bool colored =
                    boundaryMask.image.pixelColor(x, y) == QColor(85, 0, 255);
            if (filled != colored) {
                std::fprintf(stderr,
                             "OpenCV fillPoly boundary mismatch at %d,%d\n", x,
                             y);
                return 1;
            }
        }
    }
#endif
    QImage island(16, 16, QImage::Format_ARGB32);
    island.fill(Qt::transparent);
    island.setPixel(4, 4, qRgba(255, 255, 255, 255));
    island.setPixel(12, 12, qRgba(255, 255, 255, 255));
    QByteArray islandPng;
    QBuffer islandBuffer(&islandPng);
    if (!islandBuffer.open(QIODevice::WriteOnly) ||
        !island.save(&islandBuffer, "PNG"))
        return 1;
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
    if (!exactExport.error.isEmpty() ||
        exactExport.image.pixelColor(4, 4) != QColor(85, 0, 255) ||
        exactExport.image.pixelColor(12, 12) != QColor(85, 0, 255) ||
        exactExport.image.pixelColor(8, 8) != QColor(0, 0, 0)) {
        std::fprintf(stderr, "Exact mask export lost a hole or island\n");
        return 1;
    }
    const AnnotationImport exactImport =
            importMask(exactExport.image, rgbMap, QStringLiteral("mask.png"),
                       QSize(16, 16));
    const QJsonArray exactShapes =
            exactImport.document.value(QStringLiteral("shapes")).toArray();
    if (!exactImport.error.isEmpty() || exactShapes.size() != 1 ||
        exactShapes.first()
                .toObject()
                .value(QStringLiteral("acloudviewer_mask_png"))
                .toString()
                .isEmpty() ||
        convertMask(exactImport.document, rgbMap).image != exactExport.image) {
        std::fprintf(stderr, "Exact mask import round-trip failed\n");
        return 1;
    }
    const AnnotationImport importedRgbMask = importMask(
            rgbMask.image, rgbMap, QStringLiteral("mask.png"), QSize(16, 16));
    if (!importedRgbMask.error.isEmpty() || importedRgbMask.count != 1 ||
        importedRgbMask.document.value(QStringLiteral("shapes"))
                        .toArray()
                        .size() != 1 ||
        importedRgbMask.document.value(QStringLiteral("shapes"))
                        .toArray()[0]
                        .toObject()
                        .value(QStringLiteral("label"))
                        .toString() != QStringLiteral("cat")) {
        std::fprintf(stderr, "RGB mask import failed\n");
        return 1;
    }
    const QJsonObject grayMap{
            {QStringLiteral("type"), QStringLiteral("grayscale")},
            {QStringLiteral("colors"),
             QJsonObject{{QStringLiteral("cat"), 8},
                         {QStringLiteral("dog"), 3}}}};
    const MaskConversion grayMask = convertMask(maskDocument, grayMap);
    if (!grayMask.error.isEmpty() ||
        grayMask.image.format() != QImage::Format_Grayscale8 ||
        grayMask.image.pixelColor(7, 7) != QColor(3, 3, 3)) {
        std::fprintf(stderr, "Grayscale mask conversion failed\n");
        return 1;
    }
    const AnnotationImport importedGrayMask = importMask(
            grayMask.image, grayMap, QStringLiteral("mask.png"), QSize(16, 16));
    if (!importedGrayMask.error.isEmpty() || importedGrayMask.count != 2 ||
        importMask(grayMask.image, grayMap, QStringLiteral("mask.png"),
                   QSize(10, 10))
                .error.isEmpty()) {
        std::fprintf(stderr,
                     "Grayscale mask import or dimension guard failed\n");
        return 1;
    }
    QJsonObject badMap = rgbMap;
    badMap.insert(QStringLiteral("label_priority"),
                  QJsonObject{{QStringLiteral("unknown"), 1}});
    if (convertMask(maskDocument, badMap).error.isEmpty()) {
        std::fprintf(stderr, "Unknown mask priority was accepted\n");
        return 1;
    }
    document.remove(QStringLiteral("imageWidth"));
    if (convertYolo(document, classes, YoloFormat::Boxes).error.isEmpty()) {
        std::fprintf(stderr, "Invalid dimensions were accepted\n");
        return 1;
    }
    document.insert(QStringLiteral("imageWidth"), 100);
    document.insert(QStringLiteral("shapes"), QJsonArray{oriented});
    QJsonObject inBounds = oriented;
    inBounds.insert(QStringLiteral("points"),
                    QJsonArray{QJsonArray{10, 20}, QJsonArray{50, 20},
                               QJsonArray{50, 60}, QJsonArray{10, 60}});
    document.insert(QStringLiteral("shapes"), QJsonArray{inBounds});
    if (convertDota(document).text != "10 20 50 20 50 60 10 60 bus 0\n") {
        std::fprintf(stderr, "DOTA conversion failed\n");
        return 1;
    }
    const AnnotationImport importedDota =
            importDota(convertDota(document).text, QStringLiteral("image.png"),
                       QSize(100, 100));
    const QJsonObject dotaShape =
            importedDota.document.value(QStringLiteral("shapes"))
                    .toArray()
                    .at(0)
                    .toObject();
    if (!importedDota.error.isEmpty() || importedDota.count != 1 ||
        dotaShape.value(QStringLiteral("shape_type")).toString() !=
                QStringLiteral("rotation") ||
        dotaShape.value(QStringLiteral("direction")).toDouble() != 0.0 ||
        importDota("10 20 50 20 50 60 10 60 bus nope\n",
                   QStringLiteral("image.png"), QSize(100, 100))
                .error.isEmpty()) {
        std::fprintf(stderr, "DOTA import round-trip failed\n");
        return 1;
    }
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
    if (!cocoBoxes.error.isEmpty() || !cocoSegments.error.isEmpty() ||
        cocoBoxes.count != 1 || cocoSegments.count != 1 ||
        boxAnnotations[0]
                        .toObject()
                        .value(QStringLiteral("category_id"))
                        .toInt() != 2 ||
        segmentAnnotations[0]
                        .toObject()
                        .value(QStringLiteral("category_id"))
                        .toInt() != 1 ||
        segmentAnnotations[0]
                        .toObject()
                        .value(QStringLiteral("segmentation"))
                        .toArray()
                        .size() != 2 ||
        cocoSegments.document.value(QStringLiteral("categories"))
                        .toArray()
                        .size() != 3 ||
        cocoSegments.document.value(QStringLiteral("type")).toString() !=
                QStringLiteral("instances")) {
        std::fprintf(stderr, "COCO conversion failed\n");
        return 1;
    }
    const CocoImport importedCocoBox = importCoco(
            QJsonDocument(cocoBoxes.document).toJson(), CocoFormat::Boxes);
    const CocoImport importedCocoSeg =
            importCoco(QJsonDocument(cocoSegments.document).toJson(),
                       CocoFormat::Segments);
    const QJsonArray importedSegments =
            importedCocoSeg.documents.value(QStringLiteral("image.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray();
    if (!importedCocoBox.error.isEmpty() || importedCocoBox.count != 1 ||
        !importedCocoSeg.error.isEmpty() || importedCocoSeg.count != 2 ||
        importedSegments.size() != 2 ||
        importedSegments[0].toObject().value(QStringLiteral("group_id")) !=
                importedSegments[1].toObject().value(
                        QStringLiteral("group_id")) ||
        importCoco("{\"categories\":[],\"images\":[],\"annotations\":[{}]}",
                   CocoFormat::Boxes)
                .error.isEmpty()) {
        std::fprintf(stderr, "COCO image import round-trip failed\n");
        return 1;
    }
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
    for (const bool compressed : {false, true}) {
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
        if (!importedRle.error.isEmpty() || importedRle.count != 1 ||
            importedShapes.size() != 1) {
            std::fprintf(stderr, "COCO RLE import failed: %s\n",
                         importedRle.error.toUtf8().constData());
            return 1;
        }
        const QJsonObject shape = importedShapes.at(0).toObject();
        const QImage restored = QImage::fromData(
                QByteArray::fromBase64(
                        shape.value(QStringLiteral("acloudviewer_mask_png"))
                                .toString()
                                .toLatin1()),
                "PNG");
        if (restored.isNull() || restored.pixelColor(2, 2).alpha() == 0 ||
            restored.pixelColor(3, 3).alpha() != 0 ||
            restored.pixelColor(6, 6).alpha() == 0) {
            std::fprintf(stderr, "COCO RLE import failed\n");
            return 1;
        }
        const CocoConversion exportedRle = convertCoco(
                {{QStringLiteral("rle.png"),
                  importedRle.documents.value(QStringLiteral("rle.png"))}},
                {QStringLiteral("ring")}, true);
        const QJsonArray exportedAnnotations =
                exportedRle.document.value(QStringLiteral("annotations"))
                        .toArray();
        if (!exportedRle.error.isEmpty() || exportedRle.count != 1 ||
            exportedAnnotations.size() != 1 ||
            !exportedAnnotations.first()
                     .toObject()
                     .value(QStringLiteral("segmentation"))
                     .isObject()) {
            std::fprintf(stderr, "COCO exact mask RLE export failed\n");
            return 1;
        }
        const CocoImport roundTrip =
                importCoco(QJsonDocument(exportedRle.document).toJson(),
                           CocoFormat::Segments);
        const QJsonArray roundTripShapes =
                roundTrip.documents.value(QStringLiteral("rle.png"))
                        .value(QStringLiteral("shapes"))
                        .toArray();
        if (!roundTrip.error.isEmpty() || roundTripShapes.size() != 1) {
            std::fprintf(stderr, "COCO exact mask RLE round-trip failed\n");
            return 1;
        }
        const QImage roundTripMask = QImage::fromData(
                QByteArray::fromBase64(
                        roundTripShapes.first()
                                .toObject()
                                .value(QStringLiteral("acloudviewer_mask_png"))
                                .toString()
                                .toLatin1()),
                "PNG");
        if (roundTripMask.size() != QSize(rleWidth, rleHeight)) {
            std::fprintf(stderr, "COCO exact mask RLE size changed\n");
            return 1;
        }
        for (int x = 0; x < rleWidth; ++x)
            for (int y = 0; y < rleHeight; ++y)
                if ((restored.pixelColor(x, y).alpha() > 0) !=
                    (roundTripMask.pixelColor(x, y).alpha() > 0)) {
                    std::fprintf(stderr,
                                 "COCO exact mask RLE pixels changed\n");
                    return 1;
                }
    }
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
    const QJsonObject trackedShape =
            importedMot.documents.value(QStringLiteral("frame_000001.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray()[0]
                    .toObject();
    if (!mot.error.isEmpty() || mot.count != 1 ||
        mot.groundTruth != "1,9,10,20,40,40,1,1,1\n" ||
        mot.detections != "1,-1,10,20,40,40,1,-1,-1,-1\n" ||
        !mot.sequence.contains("seqlength = 2") ||
        !importedMot.error.isEmpty() || importedMot.count != 1 ||
        trackedShape.value(QStringLiteral("group_id")).toInt() != 9 ||
        importMot("1,9,10,20,40,40,1,99,1\n", classes, motImages)
                .error.isEmpty()) {
        std::fprintf(stderr, "MOT round-trip failed\n");
        return 1;
    }
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
    if (!mots.error.isEmpty() || mots.count != 1 ||
        mots.groundTruth != "1 12 0 100 100 [[0, 0], [50, 0], [50, 99]]\n" ||
        !mots.sequence.contains("name = MOTS") ||
        !mots.sequence.contains("seqlength = 2") ||
        !mots.detections.isEmpty()) {
        std::fprintf(stderr, "MOTS polygon export failed\n");
        return 1;
    }
    QJsonObject invalidPolygon = trackedPolygon;
    invalidPolygon.insert(QStringLiteral("points"),
                          QJsonArray{QJsonArray{1, 2}});
    const QVector<MotImage> invalidMots{
            {QStringLiteral("frame_000001.png"),
             QSize(100, 100),
             {{QStringLiteral("shapes"), QJsonArray{invalidPolygon}}}}};
    if (convertMots(invalidMots, classes, QStringLiteral("mots"))
                .error.isEmpty()) {
        std::fprintf(stderr, "Invalid MOTS polygon was accepted\n");
        return 1;
    }
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
    if (!vlm.error.isEmpty() || vlm.count != 1 || importedVlm.count != 1 ||
        importedVlm.documents.size() != 2 || vlmShapes.size() != 1 ||
        vlmShapes[0].toObject().value(QStringLiteral("label")).toString() !=
                QStringLiteral("bus") ||
        !vlm.annotations.contains("images/image.png") ||
        importedVlm.documents.value(QStringLiteral("next.png"))
                        .value(QStringLiteral("shapes"))
                        .toArray()
                        .size() != 0 ||
        importVlm("{bad json}\n").error.isEmpty() ||
        importVlm(vlm.annotations + vlm.annotations).error.isEmpty()) {
        std::fprintf(stderr, "VLM-R1 OVD round-trip failed\n");
        return 1;
    }
    const QByteArray repairedVlm(
            "{\"image\":\"image.png\",\"conversations\":[{},"
            "{\"value\":\"<answer>[{bbox_2d: [1, 2, 5, 6], label: "
            "'bus',},]</answer>\"}]}\n");
    const VlmImport repairedImport = importVlm(repairedVlm);
    if (!repairedImport.error.isEmpty() ||
        repairedImport.documents.value(QStringLiteral("image.png"))
                        .value(QStringLiteral("shapes"))
                        .toArray()
                        .size() != 1) {
        std::fprintf(stderr, "VLM-R1 repaired answer failed\n");
        return 1;
    }
    const OdvgConversion odvg = convertOdvg(odvgImages, classes);
    const OdvgImport importedOdvg = importOdvg(odvg.annotations);
    const QJsonArray odvgShapes =
            importedOdvg.documents.value(QStringLiteral("image.png"))
                    .value(QStringLiteral("shapes"))
                    .toArray();
    if (!odvg.error.isEmpty() || odvg.count != 1 ||
        !QJsonDocument::fromJson(odvg.labelMap).isObject() ||
        !importedOdvg.error.isEmpty() || importedOdvg.count != 1 ||
        importedOdvg.documents.size() != 2 || odvgShapes.size() != 1 ||
        odvgShapes[0].toObject().value(QStringLiteral("label")).toString() !=
                QStringLiteral("bus") ||
        importOdvg(odvg.annotations + odvg.annotations.split('\n').first() +
                   '\n')
                .error.isEmpty() ||
        importOdvg("{\"filename\":\"x.png\"}\n").error.isEmpty()) {
        std::fprintf(stderr, "ODVG round-trip failed\n");
        return 1;
    }
    return 0;
}
