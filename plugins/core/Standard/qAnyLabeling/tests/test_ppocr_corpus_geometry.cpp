// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Gap #3 corpus parity: the PaddleOCR TableRec val split (100 real-document
// pages) drives a full round trip through the PPOCR det adapter. Every cell
// must survive as the exact source integer corners with the exact
// transcription, and the rectangle classification must stay exact — no
// inference, service or GUI acceptance is inferred from this geometry gate.

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

#include "AnnotationExporter.h"
#include "AnnotationPpocr.h"
#include "ecvTestDataRepository.h"

namespace {
QString corpusPath() {
    return QDir(ecvTestDataRepository::extractPath(
                        ecvTestDataRepository::Dataset::Ppocr))
            .filePath(QStringLiteral("table_rec/val.txt"));
}

struct CorpusLine {
    QString fileName;
    QJsonArray cells;
};

QVector<CorpusLine> loadCorpus(QString* error) {
    QVector<CorpusLine> lines;
    QFile file(corpusPath());
    if (!file.open(QIODevice::ReadOnly)) {
        *error = QStringLiteral("Cannot open %1").arg(corpusPath());
        return lines;
    }
    const QList<QByteArray> rows = file.readAll().split('\n');
    for (const QByteArray& row : rows) {
        if (row.trimmed().isEmpty()) continue;
        QJsonParseError parse;
        const QJsonDocument data =
                QJsonDocument::fromJson(row.trimmed(), &parse);
        if (parse.error != QJsonParseError::NoError || !data.isObject()) {
            *error = QStringLiteral("Invalid corpus line: %1")
                             .arg(QString::fromUtf8(row.left(60)));
            return lines;
        }
        const QJsonObject page = data.object();
        CorpusLine item;
        item.fileName =
                QFileInfo(page.value(QStringLiteral("filename")).toString())
                        .fileName();
        // TableRec rows nest the ground-truth cells under "html".
        item.cells = page.value(QStringLiteral("html"))
                             .toObject()
                             .value(QStringLiteral("cells"))
                             .toArray();
        if (item.fileName.isEmpty() || item.cells.isEmpty()) {
            *error = QStringLiteral("Corpus page without cells: %1")
                             .arg(item.fileName);
            return lines;
        }
        lines.append(item);
    }
    return lines;
}

QByteArray ppocrLine(const CorpusLine& line) {
    // TableRec cells carry tokens (per-character) and one bbox quad; the
    // PPOCR det ground-truth format joins the tokens as the transcription.
    QJsonArray items;
    for (const QJsonValue& value : line.cells) {
        const QJsonObject cell = value.toObject();
        QStringList tokens;
        for (const QJsonValue& token :
             cell.value(QStringLiteral("tokens")).toArray())
            tokens.append(token.toString());
        QJsonArray points;
        const QJsonArray quad =
                cell.value(QStringLiteral("bbox")).toArray().at(0).toArray();
        for (const QJsonValue& corner : quad) points.append(corner);
        items.append(QJsonObject{
                {QStringLiteral("transcription"), tokens.join(QString())},
                {QStringLiteral("points"), points},
                {QStringLiteral("difficult"), false}});
    }
    return line.fileName.toUtf8() + '\t' +
           QJsonDocument(items).toJson(QJsonDocument::Compact) + '\n';
}

bool exactPoints(const QJsonArray& produced, const QJsonArray& source) {
    if (produced.size() != source.size()) return false;
    for (int i = 0; i < source.size(); ++i) {
        const QJsonArray producedPoint = produced.at(i).toArray();
        const QJsonArray sourcePoint = source.at(i).toArray();
        if (producedPoint.size() != 2 || sourcePoint.size() != 2) return false;
        // The adapter must keep the source integers exactly (truncation
        // toward zero, matching upstream list(map(int, p))).
        if (producedPoint.at(0).toDouble() != sourcePoint.at(0).toDouble() ||
            producedPoint.at(1).toDouble() != sourcePoint.at(1).toDouble())
            return false;
    }
    return true;
}
}  // namespace

TEST(PpocrCorpusGeometry, ImportKeepsExactIntegerCornersAndTranscription) {
    QString error;
    const QVector<CorpusLine> corpus = loadCorpus(&error);
    if (!error.isEmpty()) GTEST_SKIP() << error.toUtf8().constData();
    ASSERT_EQ(corpus.size(), 100);

    int verifiedCells = 0;
    int rectangles = 0;
    for (const CorpusLine& line : corpus) {
        const PpocrImport imported =
                importPpocr(ppocrLine(line), PpocrMode::Rec);
        ASSERT_TRUE(imported.error.isEmpty())
                << line.fileName.toUtf8().constData() << ": "
                << imported.error.toUtf8().constData();
        const QJsonObject document = imported.documents.value(line.fileName);
        ASSERT_FALSE(document.isEmpty());
        const QJsonArray shapes =
                document.value(QStringLiteral("shapes")).toArray();
        ASSERT_EQ(shapes.size(), line.cells.size());
        for (int i = 0; i < line.cells.size(); ++i) {
            const QJsonObject cell = line.cells.at(i).toObject();
            const QJsonObject shape = shapes.at(i).toObject();
            EXPECT_TRUE(
                    exactPoints(shape.value(QStringLiteral("points")).toArray(),
                                cell.value(QStringLiteral("bbox"))
                                        .toArray()
                                        .at(0)
                                        .toArray()))
                    << line.fileName.toUtf8().constData() << " cell " << i;
            QStringList tokens;
            for (const QJsonValue& token :
                 cell.value(QStringLiteral("tokens")).toArray())
                tokens.append(token.toString());
            EXPECT_EQ(shape.value(QStringLiteral("description")).toString(),
                      tokens.join(QString()))
                    << line.fileName.toUtf8().constData() << " cell " << i;
            // Exact rectangle classification on the axis-aligned corpus
            // quads (upstream rectangle-or-quadrilateral decision).
            EXPECT_TRUE(shape.value(QStringLiteral("shape_type")).toString() ==
                                QStringLiteral("rectangle") ||
                        shape.value(QStringLiteral("shape_type")).toString() ==
                                QStringLiteral("quadrilateral"))
                    << line.fileName.toUtf8().constData() << " cell " << i;
            if (shape.value(QStringLiteral("shape_type")).toString() ==
                QStringLiteral("rectangle"))
                ++rectangles;
            ++verifiedCells;
        }
    }
    EXPECT_GT(verifiedCells, 1000);
    EXPECT_GT(rectangles, verifiedCells / 2);
}

TEST(PpocrCorpusGeometry, ExportRoundTripRestoresTheCorpus) {
    QString error;
    const QVector<CorpusLine> corpus = loadCorpus(&error);
    if (!error.isEmpty()) GTEST_SKIP() << error.toUtf8().constData();
    ASSERT_EQ(corpus.size(), 100);

    // Import every page once, then convert the resulting XLabel documents
    // back to the PPOCR det text and require byte-level integer equality
    // with the corpus ground truth.
    QByteArray allLines;
    for (const CorpusLine& line : corpus) allLines += ppocrLine(line);
    const PpocrImport imported = importPpocr(allLines, PpocrMode::Rec);
    ASSERT_TRUE(imported.error.isEmpty())
            << imported.error.toUtf8().constData();

    QVector<CocoImage> images;
    for (const CorpusLine& line : corpus) {
        CocoImage image;
        image.fileName = line.fileName;
        image.document = imported.documents.value(line.fileName);
        images.append(image);
    }
    const PpocrConversion converted = convertPpocr(images, PpocrMode::Rec);
    ASSERT_TRUE(converted.error.isEmpty())
            << converted.error.toUtf8().constData();

    int verifiedCells = 0;
    const QList<QByteArray> rows = converted.text.split('\n');
    ASSERT_EQ(rows.size(), corpus.size() + 1);  // trailing newline
    for (int r = 0; r < corpus.size(); ++r) {
        const int tab = rows[r].indexOf('\t');
        ASSERT_GT(tab, 0);
        ASSERT_EQ(rows[r].left(tab), corpus[r].fileName.toUtf8());
        QJsonParseError parse;
        const QJsonDocument data =
                QJsonDocument::fromJson(rows[r].mid(tab + 1), &parse);
        ASSERT_EQ(parse.error, QJsonParseError::NoError);
        ASSERT_TRUE(data.isArray());
        const QJsonArray annotations = data.array();
        ASSERT_EQ(annotations.size(), corpus[r].cells.size());
        for (int i = 0; i < corpus[r].cells.size(); ++i) {
            const QJsonObject cell = corpus[r].cells.at(i).toObject();
            const QJsonObject produced = annotations.at(i).toObject();
            EXPECT_TRUE(exactPoints(
                    produced.value(QStringLiteral("points")).toArray(),
                    cell.value(QStringLiteral("bbox"))
                            .toArray()
                            .at(0)
                            .toArray()))
                    << corpus[r].fileName.toUtf8().constData() << " cell " << i;
            QStringList tokens;
            for (const QJsonValue& token :
                 cell.value(QStringLiteral("tokens")).toArray())
                tokens.append(token.toString());
            EXPECT_EQ(
                    produced.value(QStringLiteral("transcription")).toString(),
                    tokens.join(QString()))
                    << corpus[r].fileName.toUtf8().constData() << " cell " << i;
            ++verifiedCells;
        }
    }
    EXPECT_EQ(verifiedCells, imported.count);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
