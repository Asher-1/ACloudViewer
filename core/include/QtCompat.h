// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Qt5/Qt6 Compatibility Layer
//
// This header provides compatibility wrappers for APIs that differ between Qt5
// and Qt6. Include this header instead of directly using Qt headers for
// compatibility-sensitive APIs.
//
// USAGE:
//   1. Include this header: #include "QtCompat.h"
//   2. Replace incompatible APIs with QtCompat equivalents:
//      - QRegExp -> QtCompatRegExp or use qtCompatSplitRegex()
//      - QStringRef -> QtCompatStringRef or use qtCompatStringRef()
//      - QString::SkipEmptyParts -> QtCompat::SkipEmptyParts
//      - QFontMetrics::width() -> QTCOMPAT_FONTMETRICS_WIDTH()
//
// EXAMPLES:
//   Old: QStringList parts = str.split(QRegExp("\\s+"),
//   QString::SkipEmptyParts); New: QStringList parts = qtCompatSplitRegex(str,
//   "\\s+", QtCompat::SkipEmptyParts);
//
//   Old: static const QRegExp filter("pattern");
//        text.replace(filter, "");
//   New: static const QtCompatRegExp filter("pattern");  // Efficient: compiled
//   once
//        text.replace(filter, "");  // Direct call, no overhead
//
//   Old: int w = fm.width(text);
//   New: int w = QTCOMPAT_FONTMETRICS_WIDTH(fm, text);
//
// See QtCompat.example.cpp for more detailed usage examples.
// ----------------------------------------------------------------------------

#pragma once

#include <QString>
#include <QStringList>
#include <QtGlobal>

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6 includes
#include <QRegularExpression>
#include <QStringView>
#include <Qt>
#else
// Qt5 includes
#include <QRegExp>
#include <QStringRef>
#endif

// ----------------------------------------------------------------------------
// QRegExp / QRegularExpression Compatibility
// ----------------------------------------------------------------------------
// Qt5: QRegExp
// Qt6: QRegularExpression (QRegExp removed)
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use QRegularExpression
using QtCompatRegExp = QRegularExpression;

// Qt6 RegExp options
namespace QtCompatRegExpOption {
constexpr QRegularExpression::PatternOption CaseInsensitive =
        QRegularExpression::CaseInsensitiveOption;
constexpr QRegularExpression::PatternOption DotMatchesEverything =
        QRegularExpression::DotMatchesEverythingOption;
constexpr QRegularExpression::PatternOption Multiline =
        QRegularExpression::MultilineOption;
}  // namespace QtCompatRegExpOption

// Helper function to convert QRegExp pattern to QRegularExpression
inline QRegularExpression qtCompatRegExp(const QString& pattern) {
    return QRegularExpression(pattern);
}

// Helper function for QString::split with QRegularExpression
inline QStringList qtCompatSplit(
        const QString& str,
        const QRegularExpression& regex,
        Qt::SplitBehavior behavior = Qt::KeepEmptyParts) {
    return str.split(regex, behavior);
}
#else
// Qt5: Use QRegExp
using QtCompatRegExp = QRegExp;

// Qt5 RegExp options (using Qt namespace values)
namespace QtCompatRegExpOption {
constexpr Qt::CaseSensitivity CaseInsensitive = Qt::CaseInsensitive;
constexpr Qt::CaseSensitivity CaseSensitive = Qt::CaseSensitive;
}  // namespace QtCompatRegExpOption

// Helper function for Qt5 compatibility
inline QRegExp qtCompatRegExp(const QString& pattern) {
    return QRegExp(pattern);
}

// Helper function for QString::split with QRegExp
inline QStringList qtCompatSplit(
        const QString& str,
        const QRegExp& regex,
        QString::SplitBehavior behavior = QString::KeepEmptyParts) {
    return str.split(regex, behavior);
}
#endif

// ----------------------------------------------------------------------------
// QStringRef / QStringView Compatibility
// ----------------------------------------------------------------------------
// Qt5: QStringRef
// Qt6: QStringView (QStringRef removed)
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use QStringView
using QtCompatStringRef = QStringView;

// Helper to create QStringView from QString
inline QStringView qtCompatStringRef(const QString& str) {
    return QStringView(str);
}

// Helper to convert QStringView to QString
inline QString qtCompatStringRefToString(const QStringView& view) {
    return view.toString();
}

// Helper to create QStringView from QString with range
inline QStringView qtCompatStringRef(const QString& str, int pos, int n = -1) {
    if (n < 0) {
        return QStringView(str).mid(pos);
    }
    return QStringView(str).mid(pos, n);
}

// Helper for QString::splitRef (Qt6 uses split with QStringView)
inline QList<QStringView> qtCompatSplitRef(const QString& str, QChar sep) {
    QList<QStringView> result;
    QStringView view(str);
    qsizetype start = 0;
    while (start < view.length()) {
        qsizetype end = view.indexOf(sep, start);
        if (end < 0) {
            result.append(view.mid(start));
            break;
        }
        result.append(view.mid(start, end - start));
        start = end + 1;
    }
    return result;
}
#else
// Qt5: Use QStringRef
using QtCompatStringRef = QStringRef;

// Helper to create QStringRef from QString
inline QStringRef qtCompatStringRef(const QString& str) {
    return QStringRef(&str);
}

// Helper to convert QStringRef to QString
inline QString qtCompatStringRefToString(const QStringRef& ref) {
    return ref.toString();
}

// Helper to create QStringRef from QString with range
inline QStringRef qtCompatStringRef(const QString& str, int pos, int n = -1) {
    return QStringRef(&str, pos, n < 0 ? str.length() - pos : n);
}

// Helper for QString::splitRef (Qt5 native)
inline QVector<QStringRef> qtCompatSplitRef(const QString& str, QChar sep) {
    return str.splitRef(sep);
}
#endif

// ----------------------------------------------------------------------------
// QString::SplitBehavior / Qt::SplitBehavior Compatibility
// ----------------------------------------------------------------------------
// Qt5: QString::SkipEmptyParts, QString::KeepEmptyParts
// Qt6: Qt::SkipEmptyParts, Qt::KeepEmptyParts (moved to Qt namespace)
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use Qt namespace
namespace QtCompat {
constexpr Qt::SplitBehavior SkipEmptyParts = Qt::SkipEmptyParts;
constexpr Qt::SplitBehavior KeepEmptyParts = Qt::KeepEmptyParts;
}  // namespace QtCompat

// Helper function for QString::split with regex pattern
// This is the most common pattern in the codebase
inline QStringList qtCompatSplitRegex(
        const QString& str,
        const QString& pattern,
        Qt::SplitBehavior behavior = Qt::KeepEmptyParts) {
    return str.split(QRegularExpression(pattern), behavior);
}
#else
// Qt5: Use QString namespace
namespace QtCompat {
constexpr QString::SplitBehavior SkipEmptyParts = QString::SkipEmptyParts;
constexpr QString::SplitBehavior KeepEmptyParts = QString::KeepEmptyParts;
}  // namespace QtCompat

// Helper function for QString::split with regex pattern
inline QStringList qtCompatSplitRegex(
        const QString& str,
        const QString& pattern,
        QString::SplitBehavior behavior = QString::KeepEmptyParts) {
    return str.split(QRegExp(pattern), behavior);
}
#endif

// ----------------------------------------------------------------------------
// QFontMetrics::width() / horizontalAdvance() Compatibility
// ----------------------------------------------------------------------------
// Qt5: QFontMetrics::width() (deprecated in Qt5.11+)
// Qt6: QFontMetrics::horizontalAdvance() (width() removed)
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Only horizontalAdvance() exists
#define QTCOMPAT_FONTMETRICS_WIDTH(fm, text) (fm).horizontalAdvance(text)
#else
// Qt5: Prefer horizontalAdvance() if available (Qt5.11+), fallback to width()
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
#define QTCOMPAT_FONTMETRICS_WIDTH(fm, text) (fm).horizontalAdvance(text)
#else
#define QTCOMPAT_FONTMETRICS_WIDTH(fm, text) (fm).width(text)
#endif
#endif

// ----------------------------------------------------------------------------
// QString::replace() with QRegExp / QRegularExpression Compatibility
// ----------------------------------------------------------------------------
// Qt5: QString::replace(const QRegExp&, ...)
// Qt6: QString::replace(const QRegularExpression&, ...)
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use QRegularExpression

// Overload 1: Accept pre-compiled regex object (more efficient for repeated
// use)
inline QString& qtCompatReplace(QString& str,
                                const QRegularExpression& regex,
                                const QString& after) {
    return str.replace(regex, after);
}

inline QString qtCompatReplace(const QString& str,
                               const QRegularExpression& regex,
                               const QString& after) {
    QString result = str;
    result.replace(regex, after);
    return result;
}

// Overload 2: Accept pattern string (convenient but less efficient)
inline QString& qtCompatReplace(QString& str,
                                const QString& pattern,
                                const QString& after) {
    return str.replace(QRegularExpression(pattern), after);
}

inline QString qtCompatReplace(const QString& str,
                               const QString& pattern,
                               const QString& after) {
    QString result = str;
    result.replace(QRegularExpression(pattern), after);
    return result;
}
#else
// Qt5: Use QRegExp

// Overload 1: Accept pre-compiled regex object (more efficient for repeated
// use)
inline QString& qtCompatReplace(QString& str,
                                const QRegExp& regex,
                                const QString& after) {
    return str.replace(regex, after);
}

inline QString qtCompatReplace(const QString& str,
                               const QRegExp& regex,
                               const QString& after) {
    QString result = str;
    result.replace(regex, after);
    return result;
}

// Overload 2: Accept pattern string (convenient but less efficient)
inline QString& qtCompatReplace(QString& str,
                                const QString& pattern,
                                const QString& after) {
    return str.replace(QRegExp(pattern), after);
}

inline QString qtCompatReplace(const QString& str,
                               const QString& pattern,
                               const QString& after) {
    QString result = str;
    result.replace(QRegExp(pattern), after);
    return result;
}
#endif

// ----------------------------------------------------------------------------
// Additional Convenience Functions
// ----------------------------------------------------------------------------

// For QString::splitRef with character separator
// Usage: qtCompatSplitRefChar(str, '.')
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
inline QList<QStringView> qtCompatSplitRefChar(const QString& str, QChar sep) {
    return qtCompatSplitRef(str, sep);
}
#else
inline QVector<QStringRef> qtCompatSplitRefChar(const QString& str, QChar sep) {
    return str.splitRef(sep);
}
#endif

// ----------------------------------------------------------------------------
// QRegExp / QRegularExpression Match Compatibility
// ----------------------------------------------------------------------------
// Qt5: QRegExp::indexIn() returns match position or -1
// Qt6: QRegularExpression::match() returns QRegularExpressionMatch
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use QRegularExpression::match()
inline bool qtCompatRegExpMatch(const QRegularExpression& regex,
                                const QString& str) {
    return regex.match(str).hasMatch();
}
#else
// Qt5: Use QRegExp::indexIn()
inline bool qtCompatRegExpMatch(const QRegExp& regex, const QString& str) {
    return regex.indexIn(str) >= 0;
}
#endif
