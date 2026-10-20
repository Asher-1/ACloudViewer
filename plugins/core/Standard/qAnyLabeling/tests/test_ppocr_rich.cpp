// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QClipboard>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QMimeData>
#include <QMouseEvent>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QTemporaryDir>
#include <QTextBlock>
#include <QTextDocumentFragment>
#include <QTextEdit>
#include <QToolButton>
#include <QUrl>
#include <cstdio>

#include "PpocrRichTextEditor.h"
#include "QtCompat.h"

#define REQUIRE(condition)                                               \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                    \
        }                                                                \
    } while (false)

bool chooseHeading(PpocrRichTextEditor& editor, int level) {
    auto* button = editor.findChild<QToolButton*>("PpocrRichHeading");
    auto* popup = editor.findChild<QWidget*>("PpocrRichHeadingDropdown");
    if (!button || !popup) return false;
    button->click();
    if (!popup->isVisible()) return false;
    auto* item = popup->findChild<QWidget*>(
            QString("PpocrRichHeadingItem%1").arg(level));
    if (!item) return false;
    QMouseEvent press(QEvent::MouseButtonPress, QPointF(5, 5), Qt::LeftButton,
                      Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(item, &press);
    QApplication::processEvents();
    return !popup->isVisible();
}

int imageFramePixels(QWidget* viewport) {
    const auto image =
            viewport->grab().toImage().convertToFormat(QImage::Format_RGB32);
    int count = 0;
    for (int y = 0; y < image.height(); ++y)
        for (int x = 0; x < image.width(); ++x) {
            const auto color = image.pixelColor(x, y);
            count += color.red() == 70 && color.green() == 88 &&
                     color.blue() == 255;
        }
    return count;
}

QJsonArray rectangle(const QRect& value) {
    return {value.x(), value.y(), value.width(), value.height()};
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QJsonArray cases;
    const QStringList inputs = {
            "plain text",
            "# Heading\n\nBody **bold** and *italic*",
            "\n\ntrim edges\n\n",
            "<p>HTML <strong>bold</strong></p>",
            "<P>Uppercase</P>",
            "<unknown>literal</unknown>",
            "<p><s>strike</s> normal</p>",
            "<p><span style='text-decoration:line-through'>strike</span></p>",
            QString::fromUtf8("中文 **粗体**\n\n日本語 and emoji 😀"),
            "",
            "   ",
            "first\n\n\nsecond",
            "<p><img src='missing.png'></p>"};
    for (const auto& input : inputs) {
        PpocrRichTextEditor editor(input);
        auto* text = editor.findChild<QTextEdit*>("PpocrRichContent");
        REQUIRE(text);
        QTextCursor cursor(text->document());
        cursor.select(QTextCursor::Document);
        bool strike = false;
        for (auto block = text->document()->begin(); block.isValid();
             block = block.next())
            for (auto it = block.begin(); !it.atEnd(); ++it)
                strike |= it.fragment().isValid() &&
                          it.fragment().charFormat().fontStrikeOut();
        cases.append(
                QJsonObject{{"input", input},
                            {"output", editor.content()},
                            {"markdown", qtCompatToMarkdown(*text->document())},
                            {"html", cursor.selection().toHtml()},
                            {"strike", strike}});
    }
    PpocrRichTextEditor editor("alpha beta");
    auto* text = editor.findChild<QTextEdit*>("PpocrRichContent");
    auto cursor = text->textCursor();
    cursor.setPosition(2);
    text->setTextCursor(cursor);
    editor.findChild<QToolButton*>("PpocrRichBold")->click();
    REQUIRE(editor.content().contains("**alpha**"));
    text->selectAll();
    editor.findChild<QToolButton*>("PpocrRichStrike")->click();
    REQUIRE(editor.content().contains("line-through"));
    REQUIRE(!editor.content().contains("StartFragment"));
    const auto strikeHtml = editor.content();
    PpocrRichTextEditor replay(strikeHtml);
    REQUIRE(replay.findChild<QTextEdit*>("PpocrRichContent")->toPlainText() ==
            "alpha beta");
    REQUIRE(replay.content().contains("line-through"));
    // The source merges both the selected cursor and the current format.
    // Qt 5 records those as two commands; preserve the source edit sequence.
    text->document()->undo();
    text->document()->undo();
    REQUIRE(!editor.content().contains("line-through"));
    auto* heading = editor.findChild<QToolButton*>("PpocrRichHeading");
    REQUIRE(heading && !heading->menu());
    REQUIRE(!heading->icon().isNull() && heading->iconSize() == QSize(20, 20));
    QJsonArray headingCases;
    for (int level = 1; level <= 6; ++level) {
        text->selectAll();
        REQUIRE(chooseHeading(editor, level));
        auto block = text->document()->begin();
        REQUIRE(qtCompatHeadingLevel(block.blockFormat()) == level);
        headingCases.append(QJsonObject{
                {"level", level},
                {"bottom", block.blockFormat().bottomMargin()},
                {"line_height", block.blockFormat().lineHeight()},
                {"line_type", block.blockFormat().lineHeightType()}});
        REQUIRE(chooseHeading(editor, level));
        REQUIRE(qtCompatHeadingLevel(text->document()->begin().blockFormat()) ==
                0);
        REQUIRE(editor.content().contains("**alpha**"));
    }
    text->selectAll();
    REQUIRE(chooseHeading(editor, 1));
    text->moveCursor(QTextCursor::End);
    QKeyEvent enter(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
    QApplication::sendEvent(text, &enter);
    REQUIRE(qtCompatHeadingLevel(text->textCursor().blockFormat()) == 0);
    REQUIRE(text->currentCharFormat().fontPointSize() == 10);
    PpocrRichTextEditor mixed("# One\n\n## Two");
    auto* mixedText = mixed.findChild<QTextEdit*>("PpocrRichContent");
    auto* mixedHeading = mixed.findChild<QToolButton*>("PpocrRichHeading");
    mixedText->selectAll();
    REQUIRE(mixedHeading->property("headingLevel").toInt() == 0);
    REQUIRE(chooseHeading(mixed, 2));
    REQUIRE(mixedHeading->property("headingLevel").toInt() == 2);
    for (auto block = mixedText->document()->begin(); block.isValid();
         block = block.next())
        REQUIRE(qtCompatHeadingLevel(block.blockFormat()) == 2);
    int saves = 0, cancels = 0;
    QObject::connect(&editor, &PpocrRichTextEditor::saveRequested,
                     [&](const QString& value) {
                         if (value == editor.content()) ++saves;
                     });
    QObject::connect(&editor, &PpocrRichTextEditor::cancelRequested,
                     [&]() { ++cancels; });
    editor.findChild<QPushButton*>("PpocrRichSave")->click();
    editor.findChild<QPushButton*>("PpocrRichCancel")->click();
    REQUIRE(saves == 1 && cancels == 1);
    QJsonArray pasteCases;
    const QJsonArray pasteInputs = {
            QJsonObject{{"text", " inline"}, {"expected", "base inline"}},
            QJsonObject{{"text", "first\nsecond"},
                        {"expected", "base\nfirst\nsecond"}},
            QJsonObject{{"text", "# pasted"}, {"expected", "base\n# pasted"}},
            QJsonObject{{"html", "<h2>Heading</h2><p>Body</p>"},
                        {"expected", "base\nHeading\nBody"}},
            QJsonObject{{"html", "<p>text<img src='clip.png'></p>"},
                        {"expected", "base\ntext"}},
            QJsonObject{{"html", "<img src='clip.png'>"},
                        {"text", "alt"},
                        {"expected", "basealt"}},
            QJsonObject{{"html", "<img src='clip.png'>"},
                        {"blocked", true},
                        {"expected", "base"}},
            QJsonObject{
                    {"image", true}, {"blocked", true}, {"expected", "base"}},
            QJsonObject{{"image", true},
                        {"text", "caption"},
                        {"expected", "basecaption"}},
            QJsonObject{{"url", "file:///tmp/PHOTO.JPEG"},
                        {"text", "ignored"},
                        {"blocked", true},
                        {"expected", "base"}},
            QJsonObject{{"url", "https://example.org/photo.png"},
                        {"text", "remote"},
                        {"expected", "baseremote"}},
            QJsonObject{{"html", "<strong>inline</strong>"},
                        {"expected", "baseinline"}},
    };
    for (const auto& input : pasteInputs) {
        auto row = input.toObject();
        PpocrRichTextEditor paste("base");
        auto* target = paste.findChild<QTextEdit*>("PpocrRichContent");
        target->moveCursor(QTextCursor::End);
        auto* mime = new QMimeData;
        if (row.contains("text")) mime->setText(row["text"].toString());
        if (row.contains("html")) mime->setHtml(row["html"].toString());
        if (row.contains("url")) mime->setUrls({QUrl(row["url"].toString())});
        if (row["image"].toBool()) {
            QImage image(2, 2, QImage::Format_RGB32);
            image.fill(Qt::red);
            mime->setImageData(image);
        }
        QApplication::clipboard()->setMimeData(mime);
        const bool allowed = target->canPaste();
        REQUIRE(allowed != row["blocked"].toBool());
        target->paste();
        REQUIRE(target->toPlainText() == row["expected"].toString());
        bool image = false;
        QJsonArray pastedLevels;
        const bool newBlock = allowed && target->document()->blockCount() > 1 &&
                              target->document()->begin().text() == "base";
        for (auto block = target->document()->begin(); block.isValid();
             block = block.next()) {
            if (!newBlock || block.blockNumber() > 0)
                pastedLevels.append(qtCompatHeadingLevel(block.blockFormat()));
            for (auto it = block.begin(); !it.atEnd(); ++it)
                image |= it.fragment().charFormat().isImageFormat();
        }
        REQUIRE(!image);
        if (row["html"].toString().startsWith("<h2>"))
            REQUIRE(pastedLevels.first().toInt() == 2);
        row.insert("allowed", allowed);
        row.insert("new_block", newBlock);
        row.insert("pasted_levels", pastedLevels);
        pasteCases.append(row);
    }
    // Enter after an image reuses an existing blank body block.
    PpocrRichTextEditor imageExit("<p><img src='missing.png'></p><p></p>");
    auto* imageText = imageExit.findChild<QTextEdit*>("PpocrRichContent");
    QTextCursor imageCursor(imageText->document());
    imageCursor.movePosition(QTextCursor::EndOfBlock);
    imageCursor.insertBlock();
    imageText->setTextCursor(QTextCursor(imageText->document()));
    const int beforeEnter = imageText->document()->blockCount();
    QApplication::sendEvent(imageText, &enter);
    REQUIRE(imageText->document()->blockCount() == beforeEnter);
    REQUIRE(imageText->textCursor().blockNumber() == 1);
    REQUIRE(imageText->currentCharFormat().fontPointSize() == 10);
    // Clicking an embedded image decorates it without moving the text cursor.
    // Repeat after scrolling; cursor rectangles must remain viewport-relative.
    QTemporaryDir imageDirectory;
    REQUIRE(imageDirectory.isValid());
    const QString imagePath = imageDirectory.path() + "/embedded.png";
    QImage embedded(80, 40, QImage::Format_RGB32);
    embedded.fill(Qt::green);
    REQUIRE(embedded.save(imagePath));
    QJsonArray imageSelectionCases;
    for (bool scroll : {false, true}) {
        PpocrRichTextEditor imageEditor("body");
        auto* edit = imageEditor.findChild<QTextEdit*>("PpocrRichContent");
        const auto prefix =
                scroll ? QString("<p>line</p>").repeated(100) : QString();
        edit->setHtml(prefix + QString("<p><img src='%1' width='80' "
                                       "height='40'></p><p>after</p>")
                                       .arg(imagePath));
        imageEditor.resize(420, 500);
        imageEditor.show();
        edit->moveCursor(QTextCursor::End);
        edit->ensureCursorVisible();
        QApplication::processEvents();
        REQUIRE(!scroll || edit->verticalScrollBar()->value() > 0);
        auto block = edit->document()->begin();
        while (block.isValid() &&
               !block.begin().fragment().charFormat().isImageFormat())
            block = block.next();
        REQUIRE(block.isValid());
        const int position = block.position();
        QTextCursor start(edit->document()), end(edit->document());
        start.setPosition(position);
        end.setPosition(position + 1);
        const auto first = edit->cursorRect(start),
                   last = edit->cursorRect(end);
        const QRectF imageRect(first.x(), first.y(), last.x() - first.x(),
                               first.height());
        REQUIRE(imageRect.width() == 80 && imageRect.height() == 40);
        const auto before = edit->toHtml();
        const int characterCount = edit->document()->characterCount();
        const int cursorPosition = edit->textCursor().position();
        REQUIRE(imageFramePixels(edit->viewport()) == 0);
        QMouseEvent click(QEvent::MouseButtonPress, imageRect.center(),
                          Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(edit->viewport(), &click);
        QApplication::processEvents();
        REQUIRE(edit->textCursor().position() == cursorPosition);
        REQUIRE(edit->toHtml() == before);
        REQUIRE(imageFramePixels(edit->viewport()) > 80);
        // An ordinary text click clears the frame; selecting again and deleting
        // the image also clears stale decoration without changing saved text.
        const auto afterRect = edit->cursorRect(edit->textCursor());
        QMouseEvent outside(QEvent::MouseButtonPress,
                            QPointF(afterRect.center()), Qt::LeftButton,
                            Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(edit->viewport(), &outside);
        REQUIRE(imageFramePixels(edit->viewport()) == 0);
        QApplication::sendEvent(edit->viewport(), &click);
        REQUIRE(imageFramePixels(edit->viewport()) > 80);
        end.setPosition(position);
        end.deleteChar();
        REQUIRE(imageFramePixels(edit->viewport()) == 0);
        imageSelectionCases.append(
                QJsonObject{{"scrolled", scroll},
                            {"position", position},
                            {"character_count", characterCount},
                            {"start_rect", rectangle(first)},
                            {"end_rect", rectangle(last)},
                            {"image_width", 80},
                            {"image_rect",
                             QJsonArray{imageRect.x(), imageRect.y(),
                                        imageRect.width(), imageRect.height()}},
                            {"cursor_preserved", true},
                            {"saved_text_preserved", true},
                            {"cleared_on_text_click", true},
                            {"cleared_on_delete", true}});
    }
    QTemporaryDir previewDirectory;
    REQUIRE(previewDirectory.isValid());
    QJsonArray previewCases;
    QScrollArea outer;
    outer.resize(720, 1000);
    outer.show();
    for (const auto& size :
         {QSize(560, 440), QSize(90, 180), QSize(1000, 100)}) {
        const QString path = previewDirectory.path() + "/crop.png";
        QPixmap pixmap(size);
        pixmap.fill(Qt::red);
        REQUIRE(pixmap.save(path));
        PpocrRichTextEditor preview("caption", outer.viewport(), path);
        auto* label =
                preview.findChild<QLabel*>("PPOCRRichTextBlockEditorImage");
        REQUIRE(label && !label->isHidden());
        const auto shown = label->property("pixmap").value<QPixmap>();
        REQUIRE(!shown.isNull() && shown.width() <= 280 &&
                shown.height() <= 220);
        REQUIRE(label->contentsMargins() == QMargins(24, 24, 24, 0));
        const auto before = preview.content();
        previewCases.append(QJsonObject{{"width", size.width()},
                                        {"height", size.height()},
                                        {"preview_width", shown.width()},
                                        {"preview_height", shown.height()}});
        preview.show();
        REQUIRE(preview.content() == before);
    }
    PpocrRichTextEditor missingPreview(
            "caption", nullptr, previewDirectory.path() + "/missing.png");
    REQUIRE(missingPreview.findChild<QLabel*>("PPOCRRichTextBlockEditorImage")
                    ->isHidden());
    QJsonArray heightCases;
    for (bool withImage : {false, true}) {
        PpocrRichTextEditor tall(
                QString("paragraph\n\n").repeated(200), outer.viewport(),
                withImage ? previewDirectory.path() + "/crop.png" : QString());
        auto* edit = tall.findChild<QTextEdit*>("PpocrRichContent");
        tall.show();
        for (int height : {280, 600, 1000}) {
            outer.resize(720, height);
            QCoreApplication::processEvents();
            QCoreApplication::processEvents();
            const int imageHeight = withImage ? 28 : 0;
            const int desired =
                    qMax(72, int(edit->document()->size().height()) + 40);
            const int limit = qMax(72, int(outer.viewport()->height() * .75));
            REQUIRE(edit->height() ==
                    qMin(desired, qMax(72, limit - 32 - imageHeight -
                                                   (imageHeight ? 12 : 0))));
            heightCases.append(QJsonObject{
                    {"viewport_height", outer.viewport()->height()},
                    {"document_height", edit->document()->size().height()},
                    {"image_height", imageHeight},
                    {"actual", edit->height()}});
        }
    }
    QJsonArray styleCases;
    QJsonArray dropdownCases;
    for (bool dark : {false, true}) {
        QWidget workspace;
        auto colors = workspace.palette();
        colors.setColor(QPalette::Window, dark ? QColor("#1c1c1e") : Qt::white);
        workspace.setPalette(colors);
        PpocrRichTextEditor styled("body", &workspace);
        auto* heading = styled.findChild<QToolButton*>("PpocrRichHeading");
        QJsonObject styles{
                {"mode", dark ? "dark" : "light"},
                {"frame", styled.styleSheet()},
                {"text", styled.findChild<QTextEdit*>("PpocrRichContent")
                                 ->styleSheet()},
                {"format",
                 styled.findChild<QToolButton*>("PpocrRichBold")->styleSheet()},
                {"heading", heading->styleSheet()},
                {"save",
                 styled.findChild<QPushButton*>("PpocrRichSave")->styleSheet()},
                {"cancel", styled.findChild<QPushButton*>("PpocrRichCancel")
                                   ->styleSheet()}};
        auto* popup = styled.findChild<QWidget*>("PpocrRichHeadingDropdown");
        REQUIRE(popup && popup->windowFlags().testFlag(Qt::Popup));
        REQUIRE(popup->windowFlags().testFlag(Qt::FramelessWindowHint));
        REQUIRE(popup->windowFlags().testFlag(Qt::NoDropShadowWindowHint));
        REQUIRE(popup->testAttribute(Qt::WA_TranslucentBackground));
        heading->click();
        REQUIRE(popup->isVisible());
        REQUIRE(popup->pos() ==
                heading->mapToGlobal(QPoint(0, heading->height() + 2)));
        for (int level = 1; level <= 6; ++level) {
            auto* item = popup->findChild<QWidget*>(
                    QString("PpocrRichHeadingItem%1").arg(level));
            REQUIRE(item && item->height() == 44 &&
                    item->minimumWidth() == 130);
            REQUIRE(!item->property("selected").toBool());
            REQUIRE(item->findChild<QLabel*>()->text() ==
                    QString("H%1").arg(level));
        }
        const QString activeStyle = heading->styleSheet();
        REQUIRE(activeStyle != styles["heading"].toString());
        QKeyEvent escape(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);
        QApplication::sendEvent(popup, &escape);
        QApplication::processEvents();
        REQUIRE(!popup->isVisible() &&
                heading->property("headingLevel").toInt() == 0);
        REQUIRE(heading->styleSheet() == styles["heading"].toString());
        REQUIRE(chooseHeading(styled, 1));
        REQUIRE(heading->property("headingLevel").toInt() == 1);
        styles.insert("heading_active", heading->styleSheet());
        heading->click();
        REQUIRE(popup->isVisible());
        REQUIRE(popup->findChild<QWidget*>("PpocrRichHeadingItem1")
                        ->property("selected")
                        .toBool());
        QJsonArray rows;
        for (int level = 1; level <= 6; ++level) {
            auto* item = popup->findChild<QWidget*>(
                    QString("PpocrRichHeadingItem%1").arg(level));
            auto* label = item->findChild<QLabel*>();
            rows.append(QJsonObject{
                    {"level", level},
                    {"points", label->font().pointSize()},
                    {"selected", item->property("selected").toBool()},
                    {"style", label->styleSheet()}});
        }
        heading->click();
        QApplication::processEvents();
        REQUIRE(!popup->isVisible() &&
                heading->property("headingLevel").toInt() == 1);
        dropdownCases.append(QJsonObject{{"mode", dark ? "dark" : "light"},
                                         {"levels", 6},
                                         {"row_height", 44},
                                         {"minimum_row_width", 130},
                                         {"rows", rows},
                                         {"dismiss_escape", true},
                                         {"reopen_selected", true},
                                         {"close_button_toggle", true}});
        styleCases.append(styles);
    }
    if (argc == 2) {
        QFile output(QString::fromLocal8Bit(argv[1]));
        REQUIRE(output.open(QIODevice::WriteOnly));
        const auto bytes =
                QJsonDocument(QJsonObject{{"cases", cases},
                                          {"heading_cases", headingCases},
                                          {"paste_cases", pasteCases},
                                          {"preview_cases", previewCases},
                                          {"height_cases", heightCases},
                                          {"image_selection_cases",
                                           imageSelectionCases},
                                          {"dropdown_cases", dropdownCases},
                                          {"style_cases", styleCases}})
                        .toJson();
        REQUIRE(output.write(bytes) == bytes.size());
    }
    std::puts(
            "PASS: rich text format/heading/return/undo/save/cancel and HTML "
            "replay");
    return 0;
}
