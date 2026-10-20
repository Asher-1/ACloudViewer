// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PpocrRichTextEditor.h"

#include <QAbstractScrollArea>
#include <QAbstractTextDocumentLayout>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QRegularExpression>
#include <QShortcut>
#include <QTextBlock>
#include <QTextCursor>
#include <QTextDocumentFragment>
#include <QTextEdit>
#include <QTimer>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>
#include <functional>
#include <memory>
#include <optional>

#include "QtCompat.h"

namespace {
struct Heading {
    int points, lineHeight, bottom;
};
constexpr Heading headings[] = {{10, 0, 6},    {29, 123, 19}, {23, 135, 15},
                                {18, 135, 12}, {15, 140, 10}, {12, 150, 8},
                                {11, 157, 7}};
QIcon richIcon(QChar character, const QColor& color) {
    constexpr int size = 20;
    QPixmap pixmap(size * 2, size * 2);
    pixmap.setDevicePixelRatio(2);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
    painter.setPen(color);
    QFont font("Georgia");
    if (character == 'H') {
        font.setPixelSize(17);
        font.setWeight(QFont::Medium);
        painter.setFont(font);
        const auto primary = painter.fontMetrics();
        const int primaryWidth = primary.horizontalAdvance('T');
        QFont secondary("Georgia");
        secondary.setPixelSize(12);
        painter.setFont(secondary);
        const qreal start = (size - primaryWidth -
                             painter.fontMetrics().horizontalAdvance('T') + 2) /
                            2.;
        const qreal baseline =
                (size + primary.ascent() - primary.descent()) / 2.;
        painter.setFont(font);
        painter.drawText(QPointF(start, baseline), "T");
        painter.setFont(secondary);
        painter.drawText(QPointF(start + primaryWidth - 2, baseline), "T");
    } else {
        font.setPixelSize(18);
        font.setBold(character == 'B');
        font.setItalic(character == 'I');
        painter.setFont(font);
        painter.drawText(QRectF(0, 0, size, size), Qt::AlignCenter,
                         QString(character));
        if (character == 'S') {
            painter.setPen(QPen(color, 1.3));
            painter.drawLine(QPointF(3, size / 2.),
                             QPointF(size - 3, size / 2.));
        }
    }
    painter.end();
    return QIcon(pixmap);
}
class HeadingItem final : public QWidget {
public:
    HeadingItem(int level,
                bool dark,
                QWidget* parent,
                std::function<void(int)> selected)
        : QWidget(parent),
          m_level(level),
          m_dark(dark),
          m_selected(std::move(selected)) {
        setObjectName(QString("PpocrRichHeadingItem%1").arg(level));
        setCursor(Qt::PointingHandCursor);
        setFixedHeight(44);
        setMinimumWidth(130);
        auto* layout = new QHBoxLayout(this);
        layout->setContentsMargins(16, 0, 16, 0);
        layout->setSpacing(0);
        m_label = new QLabel(QString("H%1").arg(level), this);
        QFont font;
        font.setPointSize(headings[level].points);
        font.setWeight(QFont::DemiBold);
        m_label->setFont(font);
        layout->addWidget(m_label);
        layout->addStretch();
        setSelected(false);
    }
    void setSelected(bool selected) {
        setProperty("selected", selected);
        const QString color = selected ? QString("rgb(70, 88, 255)")
                                       : (m_dark ? "#f5f5f7" : "#1d1d1f");
        m_label->setStyleSheet(
                QString("color: %1; background: transparent; border: none;")
                        .arg(color));
    }

protected:
    bool event(QEvent* event) override {
        if (event->type() == QEvent::Enter || event->type() == QEvent::Leave) {
            m_hovered = event->type() == QEvent::Enter;
            update();
        }
        return QWidget::event(event);
    }
    void paintEvent(QPaintEvent* event) override {
        if (m_hovered) {
            QPainter painter(this);
            painter.setRenderHint(QPainter::Antialiasing);
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(m_dark ? "#3a3a3c" : "#e5e5e5"));
            painter.drawRoundedRect(rect(), 4, 4);
        }
        QWidget::paintEvent(event);
    }
    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) m_selected(m_level);
        QWidget::mousePressEvent(event);
    }

private:
    int m_level;
    bool m_dark, m_hovered = false;
    QLabel* m_label;
    std::function<void(int)> m_selected;
};
class HeadingDropdown final : public QWidget {
public:
    HeadingDropdown(bool dark,
                    QWidget* parent,
                    std::function<void(int)> selected)
        : QWidget(parent,
                  Qt::Popup | Qt::FramelessWindowHint |
                          Qt::NoDropShadowWindowHint),
          m_dark(dark) {
        setObjectName("PpocrRichHeadingDropdown");
        setAttribute(Qt::WA_TranslucentBackground);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(4, 4, 4, 4);
        layout->setSpacing(0);
        for (int level = 1; level <= 6; ++level) {
            auto* item = new HeadingItem(level, dark, this,
                                         [this, selected](int level) {
                                             selected(level);
                                             close();
                                         });
            layout->addWidget(item);
            m_items.append(item);
        }
        adjustSize();
    }
    void setCurrentLevel(int level) {
        for (int i = 0; i < m_items.size(); ++i)
            m_items[i]->setSelected(i + 1 == level);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(QPen(QColor(m_dark ? "#3a3a3c" : "#E5E5E5"), 1));
        painter.setBrush(QColor(m_dark ? "#1c1c1e" : "#ffffff"));
        painter.drawRoundedRect(QRectF(rect()).adjusted(.5, .5, -.5, -.5), 6,
                                6);
    }

private:
    bool m_dark;
    QList<HeadingItem*> m_items;
};
bool imageOnly(const QTextBlock& block) {
    bool image = false;
    for (auto it = block.begin(); !it.atEnd(); ++it) {
        const auto fragment = it.fragment();
        if (!fragment.isValid()) continue;
        if (fragment.charFormat().isImageFormat())
            image = true;
        else if (!fragment.text().trimmed().isEmpty())
            return false;
    }
    return image;
}
bool empty(const QTextBlock& block) {
    return !imageOnly(block) && block.text().trimmed().isEmpty();
}
void metrics(QTextBlockFormat& format, int level) {
    const auto& heading = headings[qBound(0, level, 6)];
    format.setTopMargin(0);
    format.setBottomMargin(heading.bottom);
    format.setLineHeight(heading.lineHeight,
                         level ? QTextBlockFormat::ProportionalHeight
                               : QTextBlockFormat::SingleHeight);
}
void headingFormats(const QTextBlock& block, int previous, int target) {
    const int previousWeight = previous > 0 ? QFont::DemiBold : QFont::Normal;
    for (auto it = block.begin(); !it.atEnd();) {
        const auto fragment = it.fragment();
        ++it;
        if (!fragment.isValid() || fragment.charFormat().isImageFormat())
            continue;
        QTextCursor part(block);
        part.setPosition(fragment.position());
        part.setPosition(fragment.position() + fragment.length(),
                         QTextCursor::KeepAnchor);
        QTextCharFormat character;
        character.setFontPointSize(headings[target].points);
        character.setFontWeight(
                fragment.charFormat().fontWeight() > previousWeight
                        ? fragment.charFormat().fontWeight()
                        : (target ? QFont::DemiBold : QFont::Normal));
        part.mergeCharFormat(character);
    }
}
// Own only a sanitized replacement; leave the borrowed clipboard data intact.
const QMimeData* sanitizedMime(const QMimeData* source,
                               std::unique_ptr<QMimeData>& replacement) {
    static const QStringList images = {"png", "jpg",  "jpeg", "bmp",
                                       "gif", "webp", "tif",  "tiff"};
    for (const auto& url : source->urls())
        if (url.isLocalFile() &&
            images.contains(QFileInfo(url.toLocalFile()).suffix().toLower()))
            return nullptr;
    const auto plainText = [&]() -> const QMimeData* {
        if (!source->hasText() || source->text().trimmed().isEmpty())
            return nullptr;
        replacement.reset(new QMimeData);
        replacement->setText(source->text());
        return replacement.get();
    };
    if (!source->hasHtml()) return source->hasImage() ? plainText() : source;
    static const QRegularExpression image(
            "<img\\b[^>]*>", QRegularExpression::CaseInsensitiveOption);
    const auto html = source->html();
    if (!source->hasImage() && !image.match(html).hasMatch()) return source;
    auto sanitized = html;
    sanitized.remove(image);
    auto visible = sanitized;
    visible.remove(QRegularExpression("<[^>]+>"));
    if (visible.trimmed().isEmpty()) return plainText();
    replacement.reset(new QMimeData);
    replacement->setHtml(sanitized);
    if (source->hasText()) replacement->setText(source->text());
    return replacement.get();
}
class RichEdit final : public QTextEdit {
public:
    using QTextEdit::QTextEdit;

protected:
    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            const int previous = m_selectedImage;
            m_selectedImage = imageAt(event->pos());
            if (previous != m_selectedImage) viewport()->update();
            if (m_selectedImage >= 0) {
                event->accept();
                return;
            }
        }
        QTextEdit::mousePressEvent(event);
    }
    void paintEvent(QPaintEvent* event) override {
        QTextEdit::paintEvent(event);
        if (m_selectedImage < 0) return;
        const auto image = imageRect(m_selectedImage);
        if (!image) {
            m_selectedImage = -1;
            return;
        }
        QPainter painter(viewport());
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setBrush(Qt::NoBrush);
        painter.setPen(QPen(QColor(70, 88, 255), 3));
        painter.drawRoundedRect(image->adjusted(-4, -4, 4, 4), 10, 10);
    }
    bool canInsertFromMimeData(const QMimeData* source) const override {
        std::unique_ptr<QMimeData> replacement;
        const auto* sanitized = sanitizedMime(source, replacement);
        return sanitized && QTextEdit::canInsertFromMimeData(sanitized);
    }
    void insertFromMimeData(const QMimeData* source) override {
        std::unique_ptr<QMimeData> replacement;
        const auto* sanitized = sanitizedMime(source, replacement);
        if (!sanitized) return;
        QList<int> levels;
        if (sanitized->hasHtml()) {
            static const QRegularExpression tags(
                    "<(h[1-6]|p|div|li|blockquote)\\b",
                    QRegularExpression::CaseInsensitiveOption);
            auto matches = tags.globalMatch(sanitized->html());
            while (matches.hasNext()) {
                const auto tag = matches.next().captured(1).toLower();
                levels.append(tag.startsWith('h') ? tag.mid(1).toInt() : 0);
            }
        }
        auto cursor = textCursor();
        bool newBlock = false;
        if (cursor.positionInBlock() > 0 || !cursor.block().text().isEmpty()) {
            if (sanitized->hasHtml()) {
                static const QRegularExpression blocks(
                        "<(?:h[1-6]|p|div|ul|ol|li|blockquote|img)\\b",
                        QRegularExpression::CaseInsensitiveOption);
                newBlock = blocks.match(sanitized->html()).hasMatch();
            } else if (sanitized->hasText()) {
                auto text = sanitized->text();
                text.remove(QRegularExpression("^\\s+"));
                newBlock = text.contains('\n') ||
                           QRegularExpression(
                                   "^(#{1,6}\\s|!\\[|\\d+\\.\\s|[-*+]\\s)")
                                   .match(text)
                                   .hasMatch();
            }
        }
        if (newBlock) {
            cursor.insertBlock();
            auto format = cursor.blockFormat();
            qtCompatSetHeadingLevel(format, 0);
            metrics(format, 0);
            cursor.setBlockFormat(format);
            bodyFormat(cursor);
        }
        const int start = textCursor().block().blockNumber();
        QTextEdit::insertFromMimeData(sanitized);
        for (auto block = document()->begin(); block.isValid();
             block = block.next()) {
            QTextCursor blockCursor(block);
            auto format = block.blockFormat();
            const int level = qtCompatHeadingLevel(format);
            metrics(format, level);
            if (imageOnly(block)) {
                format.setBottomMargin(6);
                format.setLineHeight(0, QTextBlockFormat::SingleHeight);
            }
            blockCursor.mergeBlockFormat(format);
        }
        for (int index = 0; index < levels.size(); ++index) {
            if (levels[index] <= 0) continue;
            const auto block = document()->findBlockByNumber(start + index);
            if (!block.isValid()) break;
            QTextCursor blockCursor(block);
            blockCursor.movePosition(QTextCursor::EndOfBlock,
                                     QTextCursor::KeepAnchor);
            auto format = blockCursor.blockFormat();
            const int previous = qtCompatHeadingLevel(format);
            qtCompatSetHeadingLevel(format, levels[index]);
            metrics(format, levels[index]);
            blockCursor.setBlockFormat(format);
            headingFormats(block, previous, levels[index]);
        }
        cursor = textCursor();
        if (!cursor.charFormat().isImageFormat()) bodyFormat(cursor);
        setTextCursor(cursor);
    }
    void keyPressEvent(QKeyEvent* event) override {
        const bool enter =
                event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter;
        const bool reset =
                enter &&
                (qtCompatHeadingLevel(textCursor().blockFormat()) > 0 ||
                 imageOnly(textCursor().block()));
        if (enter && imageOnly(textCursor().block())) {
            const auto next = textCursor().block().next();
            if (next.isValid() && next.text().isEmpty() && !imageOnly(next)) {
                QTextCursor cursor(next);
                auto format = cursor.blockFormat();
                qtCompatSetHeadingLevel(format, 0);
                metrics(format, 0);
                cursor.setBlockFormat(format);
                bodyFormat(cursor);
                event->accept();
                return;
            }
        }
        QTextEdit::keyPressEvent(event);
        if (!enter) return;
        auto cursor = textCursor();
        auto block = cursor.blockFormat();
        if (reset) qtCompatSetHeadingLevel(block, 0);
        metrics(block, qtCompatHeadingLevel(block));
        cursor.setBlockFormat(block);
        if (reset) bodyFormat(cursor);
        setTextCursor(cursor);
    }

private:
    std::optional<QRectF> imageRect(int position) const {
        if (position + 1 >= document()->characterCount()) return std::nullopt;
        QTextCursor end(document());
        end.setPosition(position + 1);
        if (!end.charFormat().isImageFormat()) return std::nullopt;
        QTextCursor start(document());
        start.setPosition(position);
        const auto first = cursorRect(start), last = cursorRect(end);
        const qreal width = qAbs(last.y() - first.y()) < 4
                                    ? last.x() - first.x()
                                    : end.charFormat().toImageFormat().width();
        return QRectF(first.x(), first.y(), width, first.height());
    }
    int imageAt(const QPoint& point) const {
        for (auto block = document()->begin(); block.isValid();
             block = block.next()) {
            for (auto it = block.begin(); !it.atEnd(); ++it) {
                const auto fragment = it.fragment();
                if (!fragment.isValid() ||
                    !fragment.charFormat().isImageFormat())
                    continue;
                const auto image = imageRect(fragment.position());
                if (image && image->contains(point)) return fragment.position();
            }
        }
        return -1;
    }
    int m_selectedImage = -1;
    void bodyFormat(QTextCursor& cursor) {
        QTextCharFormat format;
        QFont font;
        font.setPointSize(10);
        font.setWeight(QFont::Normal);
        format.setFont(font);
        format.setForeground(palette().color(QPalette::Text));
        cursor.setCharFormat(format);
        setCurrentCharFormat(format);
        setTextCursor(cursor);
    }
};
}  // namespace

PpocrRichTextEditor::PpocrRichTextEditor(const QString& text,
                                         QWidget* parent,
                                         const QString& imagePath)
    : QFrame(parent) {
    setObjectName("PPOCRRichTextBlockEditor");
    if (parent) setPalette(parent->palette());
    const bool dark = palette().color(QPalette::Window).lightness() < 128;
    setProperty("ppocrDark", dark);
    const QString background = dark ? "#1c1c1e" : "#ffffff",
                  textColor = dark ? "#f5f5f7" : "#1d1d1f",
                  secondary = dark ? "#aeaeb2" : "#86868b",
                  border = dark ? "#3a3a3c" : "#E5E5E5",
                  panel = dark ? "#2c2c2e" : "#f5f6f7",
                  selection = dark ? "#48484a" : "rgb(241, 245, 255)";
    setStyleSheet(
            QString("QFrame#PPOCRRichTextBlockEditor { background-color: %1; "
                    "border-radius: 10px; border: 1px solid %3; }"
                    "QFrame#PPOCRRichTextBlockEditorToolbar { "
                    "background-color: %1; "
                    "border: none; border-top-left-radius: 10px; "
                    "border-top-right-radius: "
                    "10px; border-bottom: 1px solid %3; }"
                    "QFrame#PPOCRRichTextBlockEditorBody { background-color: "
                    "%2; "
                    "border: none; border-bottom-left-radius: 10px; "
                    "border-bottom-right-radius: 10px; }"
                    "QLabel#PPOCRRichTextBlockEditorImage { background: "
                    "transparent; "
                    "border: none; padding: 0px; } QLabel { color: %4; }")
                    .arg(panel, background, border, textColor));
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    auto* toolbarFrame = new QFrame(this);
    toolbarFrame->setObjectName("PPOCRRichTextBlockEditorToolbar");
    auto* toolbar = new QHBoxLayout(toolbarFrame);
    toolbar->setContentsMargins(12, 0, 12, 0);
    toolbar->setSpacing(4);
    const auto tool = [this, toolbar, selection, dark](const QString& title,
                                                       const char* name) {
        auto* button = new QToolButton(this);
        button->setObjectName(name);
        button->setIcon(
                richIcon(title.at(0), QColor(dark ? "#aeaeb2" : "#6b6f76")));
        button->setIconSize(QSize(20, 20));
        button->setCheckable(true);
        button->setFocusPolicy(Qt::NoFocus);
        button->setCursor(Qt::PointingHandCursor);
        button->setMinimumSize(32, 32);
        button->setStyleSheet(
                QString("QToolButton { background: transparent; border: none; "
                        "border-radius: 8px; padding: 4px; min-width: 32px; "
                        "min-height: 32px; } QToolButton:hover { "
                        "background-color: %1; }"
                        "QToolButton:checked { background-color: %1; }")
                        .arg(selection));
        toolbar->addWidget(button);
        return button;
    };
    m_heading = tool("H", "PpocrRichHeading");
    m_heading->setCheckable(false);
    m_bold = tool("B", "PpocrRichBold");
    m_italic = tool("I", "PpocrRichItalic");
    m_strike = tool("S", "PpocrRichStrike");
    auto* menu = new HeadingDropdown(
            dark, this, [this](int level) { applyHeading(level); });
    m_headingMenu = menu;
    menu->installEventFilter(this);
    connect(m_heading, &QToolButton::clicked, this, [this, menu]() {
        if (menu->isVisible()) {
            menu->close();
            return;
        }
        menu->setCurrentLevel(selectedHeading());
        menu->move(m_heading->mapToGlobal(QPoint(0, m_heading->height() + 2)));
        menu->show();
        updateHeadingStyle();
    });
    toolbar->addStretch();
    auto* cancel = new QPushButton(tr("Cancel"), this);
    cancel->setObjectName("PpocrRichCancel");
    auto* save = new QPushButton(tr("Save"), this);
    save->setObjectName("PpocrRichSave");
    cancel->setAutoDefault(false);
    save->setAutoDefault(false);
    cancel->setStyleSheet(
            QString("QPushButton { background: %1; color: %2; border: 1px "
                    "solid %3; "
                    "border-radius: 8px; padding: 8px 16px; font-weight: 600; }"
                    "QPushButton:hover { color: rgb(70, 88, 255); "
                    "border-color: "
                    "rgb(70, 88, 255); } QPushButton:disabled { color: %4; }")
                    .arg(background, textColor, border, secondary));
    save->setStyleSheet(
            "QPushButton { background: rgb(70, 88, 255); color: white; border: "
            "none; border-radius: 8px; padding: 8px 16px; font-weight: 600; }"
            "QPushButton:hover { background: rgb(53, 71, 235); }"
            "QPushButton:disabled { background: rgb(180, 186, 220); }");
    toolbar->addWidget(cancel);
    toolbar->addSpacing(12);
    toolbar->addWidget(save);
    layout->addWidget(toolbarFrame);
    auto* body = new QFrame(this);
    body->setObjectName("PPOCRRichTextBlockEditorBody");
    auto* bodyLayout = new QVBoxLayout(body);
    bodyLayout->setContentsMargins(0, 0, 0, 0);
    bodyLayout->setSpacing(0);
    auto* preview = new QLabel(body);
    preview->setObjectName("PPOCRRichTextBlockEditorImage");
    preview->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    preview->setContentsMargins(24, 24, 24, 0);
    preview->hide();
    bodyLayout->addWidget(preview);
    const QPixmap image(imagePath);
    if (!image.isNull()) {
        const auto scaled = image.scaled(280, 220, Qt::KeepAspectRatio,
                                         Qt::SmoothTransformation);
        m_imageHeight = scaled.height();
        preview->setPixmap(scaled);
        preview->show();
    }
    m_text = new RichEdit(body);
    m_text->setObjectName("PpocrRichContent");
    m_text->setAcceptRichText(true);
    m_text->setPlaceholderText(tr("Enter content"));
    m_text->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_text->setMinimumHeight(72);
    m_text->setStyleSheet(
            QString("QTextEdit { background-color: %1; color: %2; border: "
                    "none; "
                    "border-bottom-left-radius: 10px; "
                    "border-bottom-right-radius: 10px; "
                    "padding: 16px; font-size: 13px; "
                    "selection-background-color: "
                    "rgb(70, 88, 255); selection-color: #ffffff; }"
                    "QScrollBar:vertical { width: 6px; background: "
                    "transparent; margin: "
                    "4px 2px; } QScrollBar::handle:vertical { background: %3; "
                    "border-radius: 3px; min-height: 30px; }"
                    "QScrollBar::handle:vertical:hover { background: %4; }"
                    "QScrollBar::add-line:vertical, "
                    "QScrollBar::sub-line:vertical { "
                    "height: 0px; }")
                    .arg(background, textColor, dark ? "#48484a" : "#c1c1c1",
                         dark ? "#636366" : "#a8a8a8"));
    QFont font;
    font.setPointSize(10);
    font.setWeight(QFont::Normal);
    m_text->setFont(font);
    m_text->document()->setDefaultFont(font);
    m_text->document()->setDocumentMargin(0);
    static const QRegularExpression html(
            "</"
            "?(?:p|div|span|h[1-6]|strong|b|em|i|u|s|strike|del|img|br|ul|ol|"
            "li)\\b",
            QRegularExpression::CaseInsensitiveOption);
    if (text.contains('<') && text.contains('>') && html.match(text).hasMatch())
        m_text->setHtml(text);
    else
        qtCompatSetMarkdown(*m_text->document(), text);
    // Trim only empty edge blocks. Embedded images and internal blank lines
    // survive.
    auto* document = m_text->document();
    while (document->blockCount() > 1 && empty(document->begin())) {
        QTextCursor cursor(document);
        cursor.setPosition(document->begin().next().position(),
                           QTextCursor::KeepAnchor);
        cursor.removeSelectedText();
    }
    while (document->blockCount() > 1 && empty(document->lastBlock())) {
        QTextCursor cursor(document);
        cursor.setPosition(document->lastBlock().previous().position() +
                           document->lastBlock().previous().length() - 1);
        cursor.movePosition(QTextCursor::End, QTextCursor::KeepAnchor);
        cursor.removeSelectedText();
    }
    m_text->moveCursor(QTextCursor::End);
    bodyLayout->addWidget(m_text);
    layout->addWidget(body);
    connect(cancel, &QPushButton::clicked, this,
            &PpocrRichTextEditor::cancelRequested);
    connect(save, &QPushButton::clicked, this,
            [this]() { emit saveRequested(content()); });
    const auto bind = [this](QToolButton* button, const QKeySequence& key,
                             const std::function<void()>& action) {
        connect(button, &QToolButton::clicked, this, action);
        auto* shortcut = new QShortcut(key, this);
        shortcut->setContext(Qt::WidgetWithChildrenShortcut);
        connect(shortcut, &QShortcut::activated, this, action);
    };
    bind(m_bold, QKeySequence::Bold, [this]() {
        QTextCharFormat format;
        format.setFontWeight(m_text->currentCharFormat().fontWeight() >=
                                             QFont::Bold
                                     ? QFont::Normal
                                     : QFont::Bold);
        mergeFormat(format);
    });
    bind(m_italic, QKeySequence::Italic, [this]() {
        QTextCharFormat format;
        format.setFontItalic(!m_text->currentCharFormat().fontItalic());
        mergeFormat(format);
    });
    bind(m_strike, QKeySequence("Ctrl+Shift+X"), [this]() {
        QTextCharFormat format;
        format.setFontStrikeOut(!m_text->currentCharFormat().fontStrikeOut());
        mergeFormat(format);
    });
    auto* shortcut = new QShortcut(QKeySequence::Save, this);
    shortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(shortcut, &QShortcut::activated, save, &QPushButton::click);
    connect(m_text, &QTextEdit::currentCharFormatChanged, this,
            [this]() { updateToolbar(); });
    connect(m_text, &QTextEdit::cursorPositionChanged, this,
            [this]() { updateToolbar(); });
    connect(m_text, &QTextEdit::selectionChanged, this,
            [this]() { updateToolbar(); });
    connect(document->documentLayout(),
            &QAbstractTextDocumentLayout::documentSizeChanged, this,
            [this]() { updateHeight(); });
    updateToolbar();
    QTimer::singleShot(0, this, [this]() { updateHeight(); });
}

QString PpocrRichTextEditor::content() const {
    bool strike = false;
    for (auto block = m_text->document()->begin(); block.isValid();
         block = block.next())
        for (auto it = block.begin(); !it.atEnd(); ++it)
            strike |= it.fragment().isValid() &&
                      it.fragment().charFormat().fontStrikeOut();
    QString result;
    if (strike) {
        QTextCursor cursor(m_text->document());
        cursor.select(QTextCursor::Document);
        result = cursor.selection().toHtml().trimmed();
        const auto body =
                QRegularExpression(
                        "<body[^>]*>(.*)</body>",
                        QRegularExpression::CaseInsensitiveOption |
                                QRegularExpression::DotMatchesEverythingOption)
                        .match(result);
        if (body.hasMatch()) result = body.captured(1).trimmed();
        return result.replace("<!--StartFragment-->", "")
                .replace("<!--EndFragment-->", "")
                .trimmed();
    }
    result = qtCompatToMarkdown(*m_text->document());
    while (result.endsWith('\n')) result.chop(1);
    return result;
}
void PpocrRichTextEditor::mergeFormat(const QTextCharFormat& format) {
    auto cursor = m_text->textCursor();
    if (!cursor.hasSelection()) cursor.select(QTextCursor::WordUnderCursor);
    cursor.mergeCharFormat(format);
    m_text->mergeCurrentCharFormat(format);
    m_text->setFocus();
    updateToolbar();
}
void PpocrRichTextEditor::applyHeading(int level) {
    const auto original = m_text->textCursor();
    const int start = original.hasSelection() ? original.selectionStart()
                                              : original.position();
    const int end = original.hasSelection() ? original.selectionEnd() - 1
                                            : original.position();
    const int target = selectedHeading() == level ? 0 : level;
    auto group = original;
    group.beginEditBlock();
    for (auto block = m_text->document()->findBlock(start);
         block.isValid() && block.position() <= end; block = block.next()) {
        if (imageOnly(block)) continue;
        QTextCursor cursor(block);
        cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
        auto format = block.blockFormat();
        const int previous = qtCompatHeadingLevel(format);
        qtCompatSetHeadingLevel(format, target);
        metrics(format, target);
        cursor.setBlockFormat(format);
        // Preserve explicitly stronger fragments while changing heading size.
        headingFormats(block, previous, target);
    }
    group.endEditBlock();
    m_text->setTextCursor(original);
    updateToolbar();
    m_text->setFocus();
}
int PpocrRichTextEditor::selectedHeading() const {
    const auto cursor = m_text->textCursor();
    const int end = cursor.hasSelection() ? cursor.selectionEnd() - 1
                                          : cursor.position();
    int current = -1;
    for (auto block = m_text->document()->findBlock(cursor.selectionStart());
         block.isValid() && block.position() <= end; block = block.next()) {
        if (imageOnly(block)) continue;
        const int level = qtCompatHeadingLevel(block.blockFormat());
        if (current != -1 && current != level) return 0;
        current = level;
    }
    return current >= 1 && current <= 6 ? current : 0;
}
void PpocrRichTextEditor::updateToolbar() {
    const auto format = m_text->currentCharFormat();
    m_bold->setChecked(format.fontWeight() >= QFont::Bold);
    m_italic->setChecked(format.fontItalic());
    m_strike->setChecked(format.fontStrikeOut());
    m_heading->setProperty("headingLevel", selectedHeading());
    updateHeadingStyle();
}
void PpocrRichTextEditor::updateHeadingStyle() {
    const bool dark = property("ppocrDark").toBool();
    const bool active = selectedHeading() > 0 ||
                        (m_headingMenu && m_headingMenu->isVisible());
    const QString selection = dark ? "#48484a" : "rgb(241, 245, 255)",
                  secondary = dark ? "#aeaeb2" : "#86868b";
    m_heading->setStyleSheet(
            QString("QToolButton { background-color: %1; border: none; "
                    "border-radius: "
                    "4px; padding: 4px; min-width: 36px; min-height: 36px; "
                    "color: %2; }"
                    "QToolButton:hover { background-color: %3; color: rgb(70, "
                    "88, 255); }")
                    .arg(active ? selection : "transparent",
                         active ? QString("rgb(70, 88, 255)") : secondary,
                         selection));
}
void PpocrRichTextEditor::updateHeight() {
    int limit = 640;
    QWidget* viewport = nullptr;
    for (auto* parent = parentWidget(); parent; parent = parent->parentWidget())
        if (auto* scroll = qobject_cast<QAbstractScrollArea*>(parent)) {
            viewport = scroll->viewport();
            if (viewport->height() > 0)
                limit = qMax(72, int(viewport->height() * .75));
            break;
        }
    if (m_scrollViewport != viewport) {
        if (m_scrollViewport) m_scrollViewport->removeEventFilter(this);
        m_scrollViewport = viewport;
        if (viewport) viewport->installEventFilter(this);
    }
    const int height = qMin(
            qMax(72, int(m_text->document()->size().height()) + 40),
            qMax(72, limit - 32 - m_imageHeight - (m_imageHeight ? 12 : 0)));
    if (m_text->height() == height) return;
    m_text->setFixedHeight(height);
    emit geometryChanged();
}
bool PpocrRichTextEditor::eventFilter(QObject* watched, QEvent* event) {
    if (watched == m_headingMenu && event->type() == QEvent::Hide)
        QTimer::singleShot(0, this, [this]() { updateHeadingStyle(); });
    if (watched == m_scrollViewport && event->type() == QEvent::Resize)
        QTimer::singleShot(0, this, [this]() { updateHeight(); });
    return QFrame::eventFilter(watched, event);
}
void PpocrRichTextEditor::resizeEvent(QResizeEvent* event) {
    QFrame::resizeEvent(event);
    QTimer::singleShot(0, this, [this]() { updateHeight(); });
}
void PpocrRichTextEditor::showEvent(QShowEvent* event) {
    QFrame::showEvent(event);
    QTimer::singleShot(0, this, [this]() { updateHeight(); });
}
