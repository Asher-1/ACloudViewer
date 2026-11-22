// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvComponentStyle.h"

#include <QApplication>
#include <QColor>
#include <QIcon>
#include <QIconEngine>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QPixmapCache>
#include <QPointer>
#include <QProxyStyle>
#include <QRegularExpression>
#include <QStringList>
#include <QStyleFactory>

namespace {
QColor styleColor(const QString& value) {
    QColor color;
    color.setNamedColor(value.trimmed());
    return color.isValid() ? color : QColor();
}

void setPaletteRole(QPalette& palette,
                    QPalette::ColorRole role,
                    const QColor& color,
                    QPalette::ColorGroup group = QPalette::All) {
    if (!color.isValid()) return;
    if (group == QPalette::All) {
        palette.setColor(QPalette::Active, role, color);
        palette.setColor(QPalette::Inactive, role, color);
        palette.setColor(QPalette::Disabled, role, color);
    } else {
        palette.setColor(group, role, color);
    }
}

QPalette themePalette(const QPalette& native,
                      const QString& sheet,
                      const QString& key) {
    QPalette result = native;
    const bool dark = key == QStringLiteral("QDarkStyleSheet::Dark");
    const bool light = key == QStringLiteral("QDarkStyleSheet::Light");
    if (dark || light) {
        const QColor window(dark ? QStringLiteral("#19232D")
                                 : QStringLiteral("#FAFAFA"));
        const QColor text(dark ? QStringLiteral("#DFE1E2")
                               : QStringLiteral("#19232D"));
        setPaletteRole(result, QPalette::Window, window);
        setPaletteRole(result, QPalette::Base, window);
        setPaletteRole(result, QPalette::AlternateBase,
                       dark ? QColor(QStringLiteral("#202D39"))
                            : QColor(QStringLiteral("#F1F3F5")));
        setPaletteRole(result, QPalette::Button, window);
        setPaletteRole(result, QPalette::WindowText, text);
        setPaletteRole(result, QPalette::Text, text);
        setPaletteRole(result, QPalette::ButtonText, text);
        setPaletteRole(result, QPalette::Highlight,
                       dark ? QColor(QStringLiteral("#346792"))
                            : QColor(QStringLiteral("#9FCBFF")));
        setPaletteRole(result, QPalette::HighlightedText, text);
        setPaletteRole(result, QPalette::Text,
                       QColor(dark ? QStringLiteral("#788D9C")
                                   : QStringLiteral("#9DA9B5")),
                       QPalette::Disabled);
        setPaletteRole(result, QPalette::WindowText,
                       QColor(dark ? QStringLiteral("#788D9C")
                                   : QStringLiteral("#9DA9B5")),
                       QPalette::Disabled);
        return result;
    }

    // Parse only palette-bearing top-level rules. This deliberately ignores
    // per-widget decorative colors and never lets a previous dark palette leak
    // into a native/empty theme.
    const QString uncommented = QString(sheet).remove(
            QRegularExpression(QStringLiteral("/\\*[\\s\\S]*?\\*/")));
    const QRegularExpression block(QStringLiteral("([^{}]+)\\{([^{}]*)\\}"));
    const QRegularExpression declaration(
            QStringLiteral("([A-Za-z][A-Za-z0-9_-]*)\\s*:\\s*([^;]+)"));
    auto parseProperties = [&](const QString& body, QPalette::ColorGroup group,
                               bool widgetRule) {
        auto it = declaration.globalMatch(body);
        QColor background;
        QColor foreground;
        while (it.hasNext()) {
            const auto match = it.next();
            const QString name = match.captured(1).trimmed().toLower();
            const QColor color = styleColor(match.captured(2));
            if (!color.isValid()) continue;
            if (name == QStringLiteral("window") ||
                name == QStringLiteral("background") ||
                name == QStringLiteral("background-color")) {
                setPaletteRole(result, QPalette::Window, color, group);
                background = color;
            } else if (name == QStringLiteral("base")) {
                setPaletteRole(result, QPalette::Base, color, group);
            } else if (name == QStringLiteral("alternatebase") ||
                       name == QStringLiteral("alternate-background-color")) {
                setPaletteRole(result, QPalette::AlternateBase, color, group);
            } else if (name == QStringLiteral("windowtext") ||
                       name == QStringLiteral("foreground")) {
                setPaletteRole(result, QPalette::WindowText, color, group);
                foreground = color;
            } else if (name == QStringLiteral("text") ||
                       name == QStringLiteral("color")) {
                setPaletteRole(result, QPalette::Text, color, group);
                foreground = color;
            } else if (name == QStringLiteral("button")) {
                setPaletteRole(result, QPalette::Button, color, group);
            } else if (name == QStringLiteral("buttontext")) {
                setPaletteRole(result, QPalette::ButtonText, color, group);
            } else if (name == QStringLiteral("highlight") ||
                       name == QStringLiteral("selection-background-color")) {
                setPaletteRole(result, QPalette::Highlight, color, group);
            } else if (name == QStringLiteral("highlightedtext") ||
                       name == QStringLiteral("selection-color")) {
                setPaletteRole(result, QPalette::HighlightedText, color, group);
            } else if (name == QStringLiteral("tooltipbase")) {
                setPaletteRole(result, QPalette::ToolTipBase, color, group);
            } else if (name == QStringLiteral("tooltiptext")) {
                setPaletteRole(result, QPalette::ToolTipText, color, group);
            }
        }
        if (widgetRule) {
            if (background.isValid()) {
                setPaletteRole(result, QPalette::Base, background, group);
                setPaletteRole(result, QPalette::Button, background, group);
            }
            if (foreground.isValid()) {
                setPaletteRole(result, QPalette::WindowText, foreground, group);
                setPaletteRole(result, QPalette::ButtonText, foreground, group);
            }
        }
    };

    auto it = block.globalMatch(uncommented);
    while (it.hasNext()) {
        const auto match = it.next();
        const QString selector = match.captured(1).trimmed().toLower();
        if (selector == QStringLiteral("qpalette")) {
            parseProperties(match.captured(2), QPalette::All, true);
        } else if (selector == QStringLiteral("qpalette::disabled")) {
            parseProperties(match.captured(2), QPalette::Disabled, true);
        } else if (selector == QStringLiteral("qwidget") ||
                   selector == QStringLiteral("*")) {
            parseProperties(match.captured(2), QPalette::All, true);
        }
    }
    return result;
}

QString paletteKey() {
    const QPalette palette = QApplication::palette();
    QStringList values;
    const QPalette::ColorRole roles[] = {
            QPalette::Window,         QPalette::WindowText,
            QPalette::Base,           QPalette::AlternateBase,
            QPalette::Text,           QPalette::Button,
            QPalette::ButtonText,     QPalette::Highlight,
            QPalette::HighlightedText};
    const QPalette::ColorGroup groups[] = {QPalette::Active, QPalette::Inactive,
                                           QPalette::Disabled};
    for (const auto group : groups) {
        for (const auto role : roles) {
            values.append(
                    QString::number(palette.color(group, role).rgba(), 16));
        }
    }
    return values.join(QLatin1Char('-'));
}

static QPointer<ecvComponentStyle> componentProxy;
static QPointer<QApplication> componentOwner;
static QPalette nativePalette;
static bool nativePaletteValid = false;
static QString nativeStyleKey;

bool isStyleSheetProxy(const QStyle* style) {
    return style && QString::fromLatin1(style->metaObject()->className())
                            .contains(QStringLiteral("QStyleSheetStyle"));
}

class ComponentGlyph final : public QIconEngine {
public:
    explicit ComponentGlyph(QString glyph) : m_glyph(std::move(glyph)) {}
    QIconEngine* clone() const override { return new ComponentGlyph(*this); }
    QString key() const override {
        // Include the live palette in Qt's icon-cache identity so a retained
        // QIcon cannot keep pixels painted with the previous theme.
        return QStringLiteral("ecv.component.") + m_glyph + QLatin1Char('.') +
               paletteKey();
    }
    QSize actualSize(const QSize& size, QIcon::Mode, QIcon::State) override {
        return size;
    }
    QPixmap pixmap(const QSize& size,
                   QIcon::Mode mode,
                   QIcon::State state) override {
        QPixmap result(size);
        result.fill(Qt::transparent);
        QPainter painter(&result);
        painter.setRenderHint(QPainter::Antialiasing);
        paint(&painter, QRect(QPoint(0, 0), size), mode, state);
        painter.end();
        return result;
    }
    void paint(QPainter* painter,
               const QRect& rect,
               QIcon::Mode mode,
               QIcon::State state) override {
        if (!painter || rect.isEmpty()) return;
        const QPalette palette = QApplication::palette();
        QColor ink = palette.color(QPalette::Text);
        QColor accent = palette.color(QPalette::Highlight);
        if (mode == QIcon::Disabled) {
            ink = palette.color(QPalette::Disabled, QPalette::Text);
            accent = palette.color(QPalette::Disabled, QPalette::ButtonText);
        } else if (mode == QIcon::Selected) {
            ink = palette.color(QPalette::HighlightedText);
        } else if (mode == QIcon::Active || state == QIcon::On) {
            ink = accent;
        }
        const QColor infoColor(QStringLiteral("#2896D3"));
        const QColor warningColor(palette.color(QPalette::Window).lightnessF() <
                                                  .48
                                          ? QStringLiteral("#F3C017")
                                          : QStringLiteral("#A77600"));
        const QColor criticalColor(QStringLiteral("#C0392B"));
        painter->save();
        painter->setViewport(rect);
        painter->setWindow(QRect(0, 0, 16, 16));
        QPen pen(ink, 1.45, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        painter->setPen(pen);
        painter->setBrush(Qt::NoBrush);
        const auto path = [&](std::initializer_list<QPointF> points) {
            QPainterPath p;
            auto it = points.begin();
            if (it == points.end()) return p;
            p.moveTo(*it++);
            for (; it != points.end(); ++it) p.lineTo(*it);
            return p;
        };
        const auto folderGlyph = [&](bool opened, bool addPlus) {
            painter->setBrush(
                    QColor(accent.red(), accent.green(), accent.blue(), 46));
            painter->drawPath(path({{2.2, 4.4},
                                    {6.3, 4.4},
                                    {7.5, 5.7},
                                    {13.7, 5.7},
                                    {12.2, 12.7},
                                    {2.2, 12.7},
                                    {2.2, 4.4}}));
            painter->setBrush(Qt::NoBrush);
            if (opened) {
                painter->drawPath(path(
                        {{3.0, 7.0}, {13.5, 7.0}, {11.7, 12.7}, {2.2, 12.7}}));
                painter->setPen(QPen(accent, 1.0, Qt::SolidLine, Qt::RoundCap,
                                     Qt::RoundJoin));
                painter->drawPath(path({{4.0, 9.7}, {8.0, 9.7}}));
                painter->drawPath(path({{6.0, 7.7}, {6.0, 11.7}}));
            }
            if (addPlus) {
                painter->setPen(QPen(accent, 1.25, Qt::SolidLine, Qt::RoundCap,
                                     Qt::RoundJoin));
                painter->drawPath(path({{10.3, 10.2}, {14.0, 10.2}}));
                painter->drawPath(path({{12.15, 8.35}, {12.15, 12.05}}));
            }
        };
        const auto warningMark = [&](const QColor& color) {
            painter->setPen(QPen(color, 1.2, Qt::SolidLine, Qt::RoundCap,
                                 Qt::RoundJoin));
            painter->setBrush(
                    QColor(color.red(), color.green(), color.blue(), 28));
            painter->drawPolygon(
                    QPolygonF({{8, 1.7}, {14.25, 13.4}, {1.75, 13.4}}));
            painter->setPen(QPen(color, 1.7, Qt::SolidLine, Qt::RoundCap));
            painter->drawPath(path({{8, 5.4}, {8, 9.2}}));
            painter->setBrush(color);
            painter->setPen(Qt::NoPen);
            painter->drawEllipse(QRectF(7.3, 10.5, 1.4, 1.4));
        };
        const auto questionMark = [&](const QColor& color) {
            painter->setBrush(Qt::NoBrush);
            painter->setPen(QPen(color, 1.1, Qt::SolidLine, Qt::RoundCap,
                                 Qt::RoundJoin));
            painter->drawEllipse(QRectF(1.8, 1.8, 12.4, 12.4));
            QPainterPath mark;
            mark.moveTo(5.1, 5.5);
            mark.cubicTo(5.2, 3.6, 10.7, 3.4, 10.9, 6.1);
            mark.cubicTo(11.0, 7.7, 8.0, 7.8, 8.0, 9.7);
            painter->drawPath(mark);
            painter->setPen(Qt::NoPen);
            painter->setBrush(color);
            painter->drawEllipse(QRectF(7.25, 11.0, 1.5, 1.5));
        };
        if (m_glyph == "up")
            painter->drawPath(path({{4, 10}, {8, 6}, {12, 10}}));
        else if (m_glyph == "down")
            painter->drawPath(path({{4, 6}, {8, 10}, {12, 6}}));
        else if (m_glyph == "left")
            painter->drawPath(path({{10, 4}, {6, 8}, {10, 12}}));
        else if (m_glyph == "right")
            painter->drawPath(path({{6, 4}, {10, 8}, {6, 12}}));
        else if (m_glyph == "close")
            painter->drawPath(path({{5, 5}, {11, 11}}) +
                              path({{11, 5}, {5, 11}}));
        else if (m_glyph == "check") {
            painter->setPen(QPen(accent, 2, Qt::SolidLine, Qt::RoundCap,
                                 Qt::RoundJoin));
            painter->drawPath(path({{3.5, 8}, {6.5, 11}, {12.5, 4.5}}));
        } else if (m_glyph == "help") {
            questionMark(accent);
        } else if (m_glyph == "open" || m_glyph == "folder") {
            folderGlyph(m_glyph == "open", false);
        } else if (m_glyph == "save") {
            painter->drawRoundedRect(QRectF(2.5, 2.5, 11, 11), 1, 1);
            painter->drawRect(QRectF(5, 2.8, 5.8, 3.7));
            painter->drawRect(QRectF(4.5, 8, 7, 4.8));
        } else if (m_glyph == "apply") {
            painter->setPen(QPen(accent, 1.9, Qt::SolidLine, Qt::RoundCap,
                                 Qt::RoundJoin));
            painter->drawPath(path({{3.5, 8}, {6.5, 11}, {12.5, 4.5}}));
        } else if (m_glyph == "reset" || m_glyph == "reload") {
            painter->drawArc(QRectF(2.5, 2.5, 11, 11), 35 * 16, 285 * 16);
            painter->setBrush(accent);
            painter->setPen(Qt::NoPen);
            painter->drawPolygon(
                    QPolygonF({{11.4, 2.5}, {14, 3.1}, {12.1, 5.4}}));
        } else if (m_glyph == "trash") {
            painter->drawLine(4, 4, 12, 4);
            painter->drawLine(6, 2.7, 10, 2.7);
            painter->drawRoundedRect(QRectF(4.7, 5, 6.6, 8.7), .7, .7);
            painter->drawLine(6.7, 6.5, 6.7, 11);
            painter->drawLine(9.3, 6.5, 9.3, 11);
        } else if (m_glyph == "play") {
            painter->setPen(Qt::NoPen);
            painter->setBrush(accent);
            painter->drawPolygon(QPolygonF({{4, 2.5}, {13, 8}, {4, 13.5}}));
        } else if (m_glyph == "pause") {
            painter->setBrush(accent);
            painter->setPen(Qt::NoPen);
            painter->drawRoundedRect(QRectF(4, 3, 3, 10), .7, .7);
            painter->drawRoundedRect(QRectF(9, 3, 3, 10), .7, .7);
        } else if (m_glyph == "stop") {
            painter->setBrush(accent);
            painter->setPen(Qt::NoPen);
            painter->drawRoundedRect(QRectF(3.3, 3.3, 9.4, 9.4), 1, 1);
        } else if (m_glyph == "minimize")
            painter->drawLine(3.5, 11.5, 12.5, 11.5);
        else if (m_glyph == "maximize")
            painter->drawRect(QRectF(3.5, 3.5, 9, 9));
        else if (m_glyph == "restore") {
            painter->drawRect(QRectF(4.5, 5.5, 7.5, 7));
            painter->drawLine(6, 4, 12.5, 4);
            painter->drawLine(12.5, 4, 12.5, 10);
        } else if (m_glyph == "file") {
            painter->setBrush(
                    QColor(accent.red(), accent.green(), accent.blue(), 34));
            painter->drawRoundedRect(QRectF(3.3, 1.8, 8.7, 12.4), 1.0, 1.0);
            painter->setBrush(Qt::NoBrush);
            painter->drawPath(path({{9.0, 1.8}, {9.0, 5.1}, {12.0, 5.1}}));
            painter->drawLine(5.1, 7.1, 10.1, 7.1);
            painter->drawLine(5.1, 9.0, 10.1, 9.0);
            painter->drawLine(5.1, 10.9, 8.7, 10.9);
        } else if (m_glyph == "newfolder") {
            folderGlyph(false, true);
        } else if (m_glyph == "home") {
            painter->setBrush(
                    QColor(accent.red(), accent.green(), accent.blue(), 42));
            painter->drawPolygon(QPolygonF({{2.5, 7},
                                            {8, 2.8},
                                            {13.5, 7},
                                            {12, 8.5},
                                            {12, 13},
                                            {4, 13},
                                            {4, 8.5}}));
            painter->setBrush(Qt::NoBrush);
            painter->drawLine(7, 13, 7, 8.5);
            painter->drawLine(9, 13, 9, 8.5);
        } else if (m_glyph == "computer") {
            painter->drawRoundedRect(QRectF(2.5, 3, 11, 8), 1, 1);
            painter->drawLine(8, 11, 8, 13);
            painter->drawLine(5.5, 13, 10.5, 13);
        } else if (m_glyph == "search") {
            painter->drawEllipse(QRectF(2.5, 2.5, 8, 8));
            painter->drawLine(9, 9, 13, 13);
        } else if (m_glyph == "info") {
            QColor color = infoColor;
            if (mode == QIcon::Disabled)
                color = palette.color(QPalette::Disabled, QPalette::Text);
            painter->setPen(QPen(color, 1.35));
            painter->setBrush(
                    QColor(color.red(), color.green(), color.blue(), 22));
            painter->drawEllipse(QRectF(2, 2, 12, 12));
            painter->setPen(QPen(color, 1.55, Qt::SolidLine, Qt::RoundCap));
            painter->drawPath(path({{8, 7}, {8, 10.6}}));
            painter->setBrush(color);
            painter->setPen(Qt::NoPen);
            painter->drawEllipse(QRectF(7.3, 4.1, 1.4, 1.4));
        } else if (m_glyph == "warning") {
            warningMark(
                    mode == QIcon::Disabled
                            ? palette.color(QPalette::Disabled, QPalette::Text)
                            : warningColor);
        } else if (m_glyph == "critical") {
            QColor color =
                    mode == QIcon::Disabled
                            ? palette.color(QPalette::Disabled, QPalette::Text)
                            : criticalColor;
            painter->setPen(QPen(color, 1.0));
            painter->setBrush(color);
            painter->drawRoundedRect(QRectF(2, 2, 12, 12), 2.0, 2.0);
            painter->setPen(QPen(Qt::white, 1.7, Qt::SolidLine, Qt::RoundCap));
            painter->drawPath(path({{5, 5}, {11, 11}}) +
                              path({{11, 5}, {5, 11}}));
        } else if (m_glyph == "question") {
            questionMark(
                    mode == QIcon::Disabled
                            ? palette.color(QPalette::Disabled, QPalette::Text)
                            : infoColor);
        } else if (m_glyph == "more") {
            painter->setPen(Qt::NoPen);
            painter->setBrush(ink);
            painter->drawEllipse(QRectF(3, 7, 2, 2));
            painter->drawEllipse(QRectF(7, 7, 2, 2));
            painter->drawEllipse(QRectF(11, 7, 2, 2));
        }
        painter->restore();
    }

private:
    QString m_glyph;
};

QString glyphFor(QStyle::StandardPixmap id) {
    switch (id) {
        case QStyle::SP_ArrowUp:
            return "up";
        case QStyle::SP_ArrowDown:
            return "down";
        case QStyle::SP_ArrowLeft:
        case QStyle::SP_ArrowBack:
            return "left";
        case QStyle::SP_ArrowRight:
        case QStyle::SP_ArrowForward:
            return "right";
        case QStyle::SP_TitleBarCloseButton:
        case QStyle::SP_DockWidgetCloseButton:
        case QStyle::SP_DialogCloseButton:
            return "close";
        case QStyle::SP_TitleBarMinButton:
            return "minimize";
        case QStyle::SP_TitleBarMaxButton:
            return "maximize";
        case QStyle::SP_TitleBarNormalButton:
            return "restore";
        case QStyle::SP_DialogOkButton:
        case QStyle::SP_DialogApplyButton:
            return "apply";
        case QStyle::SP_DialogCancelButton:
        case QStyle::SP_DialogDiscardButton:
            return "close";
        case QStyle::SP_DialogSaveButton:
            return "save";
        case QStyle::SP_DialogOpenButton:
            return "open";
        case QStyle::SP_DialogResetButton:
            return "reset";
        case QStyle::SP_DialogHelpButton:
            return "help";
        case QStyle::SP_DialogYesButton:
            return "check";
        case QStyle::SP_DialogNoButton:
            return "close";
        case QStyle::SP_BrowserReload:
            return "reload";
        case QStyle::SP_BrowserStop:
            return "stop";
        case QStyle::SP_MediaPlay:
            return "play";
        case QStyle::SP_MediaPause:
            return "pause";
        case QStyle::SP_MediaStop:
            return "stop";
        case QStyle::SP_MediaSkipForward:
            return "right";
        case QStyle::SP_MediaSkipBackward:
            return "left";
        case QStyle::SP_DirIcon:
            return "folder";
        case QStyle::SP_DirOpenIcon:
            return "open";
        case QStyle::SP_FileDialogNewFolder:
            return "newfolder";
        case QStyle::SP_FileIcon:
        case QStyle::SP_FileLinkIcon:
            return "file";
        case QStyle::SP_DirLinkIcon:
            return "folder";
        case QStyle::SP_TrashIcon:
            return "trash";
        case QStyle::SP_ComputerIcon:
        case QStyle::SP_DesktopIcon:
            return "computer";
        case QStyle::SP_MessageBoxInformation:
            return "info";
        case QStyle::SP_MessageBoxWarning:
            return "warning";
        case QStyle::SP_MessageBoxCritical:
            return "critical";
        case QStyle::SP_MessageBoxQuestion:
            return "question";
        case QStyle::SP_ToolBarHorizontalExtensionButton:
        case QStyle::SP_ToolBarVerticalExtensionButton:
            return "more";
        default:
            return {};
    }
}

QString componentAssetRules(bool dark) {
    const QString mode = dark ? "dark" : "light";
    QString q;
    auto image = [&](QString file) {
        return QString(QStringLiteral("url(:/ecv/components/") + file + "-" +
                       mode + ".svg)");
    };
    QStringList checkTargets = {"QCheckBox",   "QGroupBox",   "QTreeView",
                                "QTreeWidget", "QListView",   "QListWidget",
                                "QTableView",  "QTableWidget"};
    for (const auto& target : checkTargets) {
        q += target + "::indicator { width:14px;height:14px;image:" +
             image("check-off") + ";border:none;background:transparent; }\n";
        q += target + "::indicator:unchecked { image:" + image("check-off") +
             "; }\n";
        q += target + "::indicator:checked { image:" + image("check-on") +
             "; }\n";
        q += target +
             "::indicator:indeterminate { image:" + image("check-partial") +
             "; }\n";
        q += target + "::indicator:unchecked:hover { image:" +
             image("check-off").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:checked:hover { image:" +
             image("check-on").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:indeterminate:hover { image:" +
             image("check-partial")
                     .replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:unchecked:pressed { image:" +
             image("check-off").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:checked:pressed { image:" +
             image("check-on").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:indeterminate:pressed { image:" +
             image("check-partial")
                     .replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:unchecked:focus { image:" +
             image("check-off").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:checked:focus { image:" +
             image("check-on").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:indeterminate:focus { image:" +
             image("check-partial")
                     .replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:unchecked:disabled { image:" +
             image("check-off").replace("-" + mode + ".svg", "-disabled.svg") +
             "; }\n";
        q += target + "::indicator:checked:disabled { image:" +
             image("check-on").replace("-" + mode + ".svg", "-disabled.svg") +
             "; }\n";
        q += target + "::indicator:indeterminate:disabled { image:" +
             image("check-partial")
                     .replace("-" + mode + ".svg", "-disabled.svg") +
             "; }\n";
    }
    QStringList radioTargets = {"QRadioButton"};
    for (const auto& target : radioTargets) {
        q += target + "::indicator { width:14px;height:14px;image:" +
             image("radio-off") + ";border:none;background:transparent; }\n";
        q += target + "::indicator:checked { image:" + image("radio-on") +
             "; }\n";
        q += target + "::indicator:unchecked { image:" + image("radio-off") +
             "; }\n";
        q += target + "::indicator:disabled { image:" +
             image("radio-off").replace("-" + mode + ".svg", "-disabled.svg") +
             "; }\n";
        q += target + "::indicator:unchecked:hover { image:" +
             image("radio-off").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:checked:hover { image:" +
             image("radio-on").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:unchecked:pressed { image:" +
             image("radio-off").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:checked:pressed { image:" +
             image("radio-on").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:unchecked:focus { image:" +
             image("radio-off").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:checked:focus { image:" +
             image("radio-on").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::indicator:unchecked:disabled { image:" +
             image("radio-off").replace("-" + mode + ".svg", "-disabled.svg") +
             "; }\n";
        q += target + "::indicator:checked:disabled { image:" +
             image("radio-on").replace("-" + mode + ".svg", "-disabled.svg") +
             "; }\n";
    }
    QStringList treeTargets = {"QTreeView", "QTreeWidget"};
    for (const auto& target : treeTargets) {
        q += target +
             "::branch { "
             "border:none;border-image:none;background:transparent;image:none; "
             "}\n";
        q += target + "::branch:closed:has-children { image:" + image("right") +
             ";width:12px;height:12px; }\n";
        q += target + "::branch:open:has-children { image:" + image("down") +
             ";width:12px;height:12px; }\n";
        q += target + "::branch:closed:has-children:hover { image:" +
             image("right").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::branch:open:has-children:hover { image:" +
             image("down").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::branch:closed:has-children:pressed { image:" +
             image("right").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::branch:open:has-children:pressed { image:" +
             image("down").replace("-" + mode + ".svg", "-accent.svg") +
             "; }\n";
        q += target + "::branch:closed:has-children:disabled { image:" +
             image("right").replace("-" + mode + ".svg", "-disabled.svg") +
             "; }\n";
        q += target + "::branch:open:has-children:disabled { image:" +
             image("down").replace("-" + mode + ".svg", "-disabled.svg") +
             "; }\n";
    }
    q += "QTreeView::branch:!has-children,QTreeWidget::branch:!has-children { "
         "image:none; }\n";
    QString arrow = image("down");
    q += "QMenu::right-arrow { image:" + image("right") +
         ";width:12px;height:12px; }\n";
    q += "QMenu::indicator { width:14px;height:14px;image:" +
         image("check-off") + "; }\n";
    q += "QMenu::indicator:checked { image:" + image("check-on") + "; }\n";
    q += "QMenu::indicator:non-exclusive:checked { image:" + image("check-on") +
         "; }\n";
    q += "QMenu::indicator:exclusive:checked { image:" + image("radio-on") +
         "; }\n";
    q += "QComboBox::down-arrow,QDateEdit::down-arrow,QTimeEdit::down-arrow,"
         "QDateTimeEdit::down-arrow { width:12px;height:12px;image:" +
         arrow + "; }\n";
    q += "QToolButton::menu-indicator { "
         "subcontrol-origin:padding;subcontrol-position:bottom "
         "right;width:12px;height:12px;image:" +
         arrow + "; }\n";
    q += "QComboBox::down-arrow:hover,QComboBox::down-arrow:on,QComboBox::down-"
         "arrow:focus { image:" +
         image("down").replace("-" + mode + ".svg", "-accent.svg") + "; }\n";
    q += "QAbstractSpinBox::up-button { "
         "subcontrol-origin:border;subcontrol-position:top "
         "right;width:14px;border:none;image:none; }\n";
    q += "QAbstractSpinBox::down-button { "
         "subcontrol-origin:border;subcontrol-position:bottom "
         "right;width:14px;border:none;image:none; }\n";
    q += "QAbstractSpinBox::up-arrow { width:10px;height:10px;image:" +
         image("up") + "; }\n";
    q += "QAbstractSpinBox::down-arrow { width:10px;height:10px;image:" +
         arrow + "; }\n";
    QString close = image("close"), closeHot = image("close").replace(
                                            "-" + mode + ".svg", "-accent.svg");
    q += "QTabBar::close-button,QDockWidget::close-button { "
         "width:14px;height:14px;image:" +
         close + "; }\n";
    q += "QTabBar::close-button:hover,QDockWidget::close-button:hover { "
         "image:" +
         closeHot + "; }\n";
    q += "QTabBar::close-button:pressed,QDockWidget::close-button:pressed { "
         "image:" +
         image("close").replace("-" + mode + ".svg", "-disabled.svg") + "; }\n";
    q += "QTabBar::scroller { width:16px; }\n";
    q += "QDockWidget::float-button { width:14px;height:14px;image:" +
         image("maximize") + "; }\n";
    q += "QDockWidget::float-button:hover { image:" +
         image("maximize").replace("-" + mode + ".svg", "-accent.svg") +
         "; }\n";
    q += "QToolBar::handle { image:" + image("handle") +
         ";width:8px;height:16px; }\n";
    q += "QToolBar::handle:horizontal { image:" + image("handle") +
         ";width:16px;height:8px; }\n";
    q += "QToolBar::handle:vertical { image:" + image("handle") +
         ";width:8px;height:16px; }\n";
    q += "QToolBar::extension { image:" + image("overflow") +
         ";width:16px;height:16px; }\n";
    q += "QSplitter::handle:horizontal { image:" + image("grip") +
         ";width:5px; }\n";
    q += "QSplitter::handle:vertical { image:" + image("grip") +
         ";height:5px; }\n";
    q += "QHeaderView::up-arrow { image:" + image("up") +
         ";width:12px;height:12px; }\n";
    q += "QHeaderView::down-arrow { image:" + arrow +
         ";width:12px;height:12px; }\n";
    QString left = image("left");
    QString right = image("right");
    q += "QScrollBar::left-arrow:horizontal { image:" + left + "; }\n";
    q += "QScrollBar::right-arrow:horizontal { image:" + right + "; }\n";
    q += "QScrollBar::up-arrow:vertical { image:" + image("up") + "; }\n";
    q += "QScrollBar::down-arrow:vertical { image:" + arrow + "; }\n";
    q += "QScrollBar::up-arrow:vertical:hover { image:" +
         image("up").replace("-" + mode + ".svg", "-accent.svg") + "; }\n";
    q += "QScrollBar::down-arrow:vertical:hover { image:" +
         arrow.replace("-" + mode + ".svg", "-accent.svg") + "; }\n";
    q += "QScrollBar::left-arrow:horizontal:hover { image:" +
         left.replace("-" + mode + ".svg", "-accent.svg") + "; }\n";
    q += "QScrollBar::right-arrow:horizontal:hover { image:" +
         right.replace("-" + mode + ".svg", "-accent.svg") + "; }\n";
    q += "QScrollBar::add-line,QScrollBar::sub-line { "
         "image:none;border:none;background:transparent; }\n";
    q += "QSizeGrip { image:" + image("grip") + ";width:14px;height:14px; }\n";
    return q;
}
}  // namespace

ecvComponentStyle::ecvComponentStyle(const QString& key) : QProxyStyle(key) {}

QIcon ecvComponentStyle::standardIcon(QStyle::StandardPixmap id,
                                      const QStyleOption* option,
                                      const QWidget* widget) const {
    const QString glyph = glyphFor(id);
    if (!glyph.isEmpty()) return QIcon(new ComponentGlyph(glyph));
    return QProxyStyle::standardIcon(id, option, widget);
}

bool ecvComponentStyle::install(QApplication* app, const QString& key) {
    if (!app) return false;
    QStyle* current = app->style();
    const bool stylesheetWrapper = isStyleSheetProxy(current);
    if (componentOwner == app && componentProxy &&
        (current == componentProxy.data() || stylesheetWrapper)) {
        return true;
    }

    QString styleKey = key.trimmed();
    if (styleKey == QStringLiteral("QDarkStyleSheet::Dark") ||
        styleKey == QStringLiteral("QDarkStyleSheet::Light")) {
        styleKey.clear();
    }
    if (styleKey.isEmpty() && componentOwner == app &&
        !nativeStyleKey.isEmpty())
        styleKey = nativeStyleKey;
    if (styleKey.isEmpty() && current) styleKey = current->objectName();

    QStyle* base = nullptr;
    if (!styleKey.isEmpty()) base = QStyleFactory::create(styleKey);
    if (!base) {
        styleKey = QStringLiteral("Fusion");
        base = QStyleFactory::create(styleKey);
    }
    if (!base) return false;
    const QPalette factoryPalette = base->standardPalette();
    delete base;

    auto* proxy = new ecvComponentStyle(styleKey);
    app->setStyle(proxy);
    componentProxy = proxy;
    componentOwner = app;
    nativeStyleKey = styleKey;
    nativePalette = factoryPalette;
    nativePaletteValid = true;
    app->setPalette(nativePalette);
    QPixmapCache::clear();
    return true;
}

QString ecvComponentStyle::componentRules(bool dark) {
    return componentAssetRules(dark);
}

bool ecvComponentStyle::useDarkIcons(QApplication* app,
                                     const QString& sheet,
                                     const QString& key) {
    const QPalette base = nativePaletteValid
                                  ? nativePalette
                                  : (app ? app->palette() : QPalette());
    return themePalette(base, sheet, key).color(QPalette::Window).lightnessF() <
           .48;
}

void ecvComponentStyle::applyTheme(QApplication* app,
                                   const QString& base,
                                   const QString& key) {
    if (!app) return;
    const QString begin =
            QStringLiteral("/* ECV_COMPONENT_ICON_OVERLAY_BEGIN */");
    const QString end = QStringLiteral("/* ECV_COMPONENT_ICON_OVERLAY_END */");
    QString clean = base;
    const int old = clean.indexOf(begin);
    if (old >= 0) {
        const int finish = clean.indexOf(end, old);
        clean = finish >= 0 ? clean.left(old) + clean.mid(finish + end.size())
                            : clean.left(old);
    }
    const QString styleKey = key.startsWith(QStringLiteral("QDarkStyleSheet::"))
                                     ? QString()
                                     : key;
    if (!install(app, styleKey)) return;
    const QPalette resolved = themePalette(nativePalette, clean, key);
    const bool dark = resolved.color(QPalette::Window).lightnessF() < .48;
    app->setPalette(resolved);
    QPixmapCache::clear();
    app->setStyleSheet(clean + "\n" + begin + "\n" + componentRules(dark) +
                       "\n" + end + "\n");
    QPixmapCache::clear();
}
