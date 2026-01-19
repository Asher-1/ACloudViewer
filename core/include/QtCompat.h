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
// SUPPORTED COMPATIBILITY FEATURES:
//
// 1. Regular Expressions:
//    - QRegExp (Qt5) / QRegularExpression (Qt6)
//    - Type alias: QtCompatRegExp
//    - Functions: qtCompatRegExp(), qtCompatSplit(), qtCompatSplitRegex(),
//                 qtCompatReplace(), qtCompatRegExpMatch(),
//                 qtCompatRegExpIndexIn(), qtCompatRegExpPos(),
//                 qtCompatRegExpCap(), qtCompatRegExpMatchedLength()
//    - Class: QtCompatRegExpWrapper (for QRegExp-style API)
//
// 2. String References:
//    - QStringRef (Qt5) / QStringView (Qt6)
//    - Type alias: QtCompatStringRef, QtCompatStringRefList
//    - Functions: qtCompatStringRef(), qtCompatStringRefToString(),
//                 qtCompatSplitRef(), qtCompatSplitRefChar()
//
// 3. Split Behavior:
//    - QString::SkipEmptyParts (Qt5) / Qt::SkipEmptyParts (Qt6)
//    - Namespace: QtCompat::SkipEmptyParts, QtCompat::KeepEmptyParts
//
// 4. Font Metrics:
//    - QFontMetrics::width() (Qt5) / horizontalAdvance() (Qt6)
//    - Macro: QTCOMPAT_FONTMETRICS_WIDTH(fm, text)
//
// 5. Text Codec:
//    - QTextCodec (Qt5) / QStringConverter (Qt6)
//    - Type alias: QtCompatQTextCodec
//    - Functions: qtCompatCodecForLocale(), qtCompatCodecForName()
//
// 6. Text Stream:
//    - QTextStream::endl (Qt5) / Qt::endl (Qt6)
//    - Namespace: QtCompat::endl
//    - Macro: QTCOMPAT_ENDL
//
// 7. Wheel Events:
//    - QWheelEvent::delta() / pos() (Qt5) / angleDelta() / position() (Qt6)
//    - Functions: qtCompatWheelEventDelta(), qtCompatWheelEventPos()
//
// 8. Mouse Events:
//    - QMouseEvent::pos() / globalPos() (Qt5) / position() / globalPosition()
//    (Qt6)
//    - Functions: qtCompatMouseEventPos(), qtCompatMouseEventGlobalPos(),
//                 qtCompatMouseEventPosInt(), qtCompatMouseEventGlobalPosInt()
//
// 9. Drop Events:
//    - QDropEvent::pos() (Qt5) / position() (Qt6)
//    - Functions: qtCompatDropEventPos(), qtCompatDropEventPosInt()
//
// 10. Map Operations:
//    - QMap::insertMulti() / unite() (Qt5) / removed in Qt6
//    - Functions: qtCompatMapInsertMulti(), qtCompatMapUnite()
//
// 11. QVariant Type:
//    - QVariant::type() (Qt5) / typeId() (Qt6)
//    - Type alias: QtCompatVariantType
//    - Functions: qtCompatVariantType(), qtCompatVariantIsValid(),
//                 qtCompatVariantIsNull(), qtCompatVariantIsString(),
//                 qtCompatVariantIsInt(), qtCompatVariantIsDouble(),
//                 qtCompatVariantIsBool(), qtCompatVariantIsList(),
//                 qtCompatVariantIsMap()
//
// 12. Plain Text Edit:
//    - QPlainTextEdit::setTabStopWidth() (Qt5) / setTabStopDistance() (Qt6)
//    - Function: qtCompatSetTabStopWidth()
//
// 13. Container Iterators:
//    - QSet<T>(begin, end) and QVector<T>(begin, end) constructors
//    - Qt5.0-5.14: Not supported, use manual loops
//    - Qt5.15+: Supported via iterator range constructors
//    - Qt6: Supported via iterator range constructors
//    - Functions: qtCompatQSetFromVector(), qtCompatQVectorFromSet()
//
// USAGE EXAMPLES:
//
//   Regular Expression:
//   Old: QStringList parts = str.split(QRegExp("\\s+"),
//   QString::SkipEmptyParts); New: QStringList parts = qtCompatSplitRegex(str,
//   "\\s+", QtCompat::SkipEmptyParts);
//
//   Old: static const QRegExp filter("pattern");
//        text.replace(filter, "");
//   New: static const QtCompatRegExp filter("pattern");  // Efficient: compiled
//   once
//        qtCompatReplace(text, filter, "");  // Or: text.replace(filter, "");
//
//   Font Metrics:
//   Old: int w = fm.width(text);
//   New: int w = QTCOMPAT_FONTMETRICS_WIDTH(fm, text);
//
//   String References:
//   Old: QStringRef ref = str.midRef(0, 10);
//   New: QtCompatStringRef ref = qtCompatStringRef(str, 0, 10);
//
//   Text Codec:
//   Old: QTextCodec* codec = QTextCodec::codecForLocale();
//        QString text = codec->toUnicode(data);
//   New: QtCompatQTextCodec* codec = qtCompatCodecForLocale();
//        QString text = codec->toUnicode(data);
//
//   Text Stream:
//   Old: stream << QTextStream::endl;
//   New: stream << QtCompat::endl;
//        Or: stream << QTCOMPAT_ENDL;
//
//   Mouse Events:
//   Old: QPoint pos = event->pos();
//   New: QPointF pos = qtCompatMouseEventPos(event);
//        Or for integer: QPoint pos = qtCompatMouseEventPosInt(event);
//
//   Drop Events:
//   Old: QPoint pos = event->pos();
//   New: QPointF pos = qtCompatDropEventPos(event);
//
//   QVariant Type:
//   Old: if (var.type() == QVariant::String) { ... }
//   New: if (qtCompatVariantIsString(var)) { ... }
//
// See QtCompat.example.cpp for more detailed usage examples.
// ----------------------------------------------------------------------------

#pragma once

#include <QMap>
#include <QMultiMap>
#include <QPoint>
#include <QPointF>
#include <QSet>
#include <QString>
#include <QStringList>
#include <QTextStream>
#include <QVector>
#include <QtGlobal>

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6 includes
#include <QRegularExpression>
#include <QStringView>
#include <Qt>
#else
// Qt5 includes
#include <QChar>
#include <QRegExp>
#include <QStringRef>
#include <QVector>
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
// Qt5: QStringRef, QVector<QStringRef>
// Qt6: QStringView (QStringRef removed), QList<QStringView>
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use QStringView
using QtCompatStringRef = QStringView;
using QtCompatStringRefList = QList<QStringView>;

// Helper to create QStringView from QString
inline QStringView qtCompatStringRef(const QString& str) noexcept {
    return QStringView(str);
}

// Helper to convert QStringView to QString
inline QString qtCompatStringRefToString(const QStringView& view) {
    return view.toString();
}

// Helper to create QStringView from QString with range
inline QStringView qtCompatStringRef(const QString& str,
                                     int pos,
                                     int n = -1) noexcept {
    if (n < 0) {
        return QStringView(str).mid(pos);
    }
    return QStringView(str).mid(pos, n);
}

// Helper for QString::splitRef (Qt6 uses split with QStringView)
inline QtCompatStringRefList qtCompatSplitRef(const QString& str, QChar sep) {
    QtCompatStringRefList result;
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
using QtCompatStringRefList = QVector<QStringRef>;

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
inline QtCompatStringRefList qtCompatSplitRef(const QString& str, QChar sep) {
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
inline QtCompatStringRefList qtCompatSplitRefChar(const QString& str,
                                                  QChar sep) {
    return qtCompatSplitRef(str, sep);
}

// ----------------------------------------------------------------------------
// QRegExp / QRegularExpression Match Compatibility
// ----------------------------------------------------------------------------
// Qt5: QRegExp::indexIn() returns match position or -1
// Qt6: QRegularExpression::match() returns QRegularExpressionMatch
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use QRegularExpression::match()
#include <QRegularExpressionMatch>

inline bool qtCompatRegExpMatch(const QRegularExpression& regex,
                                const QString& str) {
    return regex.match(str).hasMatch();
}

// Compatibility wrapper for QRegExp::indexIn() - returns match position or -1
inline int qtCompatRegExpIndexIn(const QRegularExpression& regex,
                                 const QString& str,
                                 int offset = 0) {
    QRegularExpressionMatch match = regex.match(str, offset);
    return match.hasMatch() ? match.capturedStart() : -1;
}

// Compatibility wrapper for QRegExp::pos() - returns position of captured group
inline int qtCompatRegExpPos(const QRegularExpressionMatch& match,
                             int nth = 0) noexcept {
    return match.capturedStart(nth);
}

// Compatibility wrapper for QRegExp::cap() - returns captured text
inline QString qtCompatRegExpCap(const QRegularExpressionMatch& match,
                                 int nth = 0) {
    return match.captured(nth);
}

// Compatibility wrapper for QRegExp::matchedLength() - returns length of match
inline int qtCompatRegExpMatchedLength(
        const QRegularExpressionMatch& match) noexcept {
    return match.capturedLength();
}

// Helper class to wrap QRegularExpression for Qt5 QRegExp-style API
// This allows code to use QRegExp-like API that works with both Qt5 and Qt6
class QtCompatRegExpWrapper {
private:
    QRegularExpression m_regex;
    mutable QRegularExpressionMatch m_lastMatch;

public:
    QtCompatRegExpWrapper() = default;

    QtCompatRegExpWrapper(const QRegularExpression& regex) : m_regex(regex) {}

    QtCompatRegExpWrapper(const QString& pattern) : m_regex(pattern) {}

    // QRegExp::indexIn() compatibility
    int indexIn(const QString& str, int offset = 0) const {
        m_lastMatch = m_regex.match(str, offset);
        return m_lastMatch.hasMatch() ? m_lastMatch.capturedStart() : -1;
    }

    // QRegExp::pos() compatibility - returns position of captured group
    int pos(int nth = 0) const noexcept {
        return m_lastMatch.capturedStart(nth);
    }

    // QRegExp::cap() compatibility - returns captured text
    QString cap(int nth = 0) const { return m_lastMatch.captured(nth); }

    // QRegExp::matchedLength() compatibility
    int matchedLength() const noexcept { return m_lastMatch.capturedLength(); }

    // Get the underlying QRegularExpression (Qt6) or convert to QRegExp (Qt5)
    const QRegularExpression& regex() const noexcept { return m_regex; }

    // Allow implicit conversion to QRegularExpression for Qt6
    operator const QRegularExpression&() const noexcept { return m_regex; }
};

#else
// Qt5: Use QRegExp::indexIn()
inline bool qtCompatRegExpMatch(const QRegExp& regex, const QString& str) {
    return regex.indexIn(str) >= 0;
}

// Qt5: Direct passthrough functions
inline int qtCompatRegExpIndexIn(const QRegExp& regex,
                                 const QString& str,
                                 int offset = 0) {
    return regex.indexIn(str, offset);
}

inline int qtCompatRegExpPos(const QRegExp& regex, int nth = 0) noexcept {
    return regex.pos(nth);
}

inline QString qtCompatRegExpCap(const QRegExp& regex, int nth = 0) {
    return regex.cap(nth);
}

inline int qtCompatRegExpMatchedLength(const QRegExp& regex) noexcept {
    return regex.matchedLength();
}

// Qt5: QRegExp wrapper that provides the same interface
class QtCompatRegExpWrapper {
private:
    mutable QRegExp m_regex;

public:
    QtCompatRegExpWrapper() = default;

    QtCompatRegExpWrapper(const QRegExp& regex) : m_regex(regex) {}

    QtCompatRegExpWrapper(const QString& pattern) : m_regex(pattern) {}

    int indexIn(const QString& str, int offset = 0) const {
        return m_regex.indexIn(str, offset);
    }

    int pos(int nth = 0) const noexcept { return m_regex.pos(nth); }

    QString cap(int nth = 0) const { return m_regex.cap(nth); }

    int matchedLength() const noexcept { return m_regex.matchedLength(); }

    const QRegExp& regex() const noexcept { return m_regex; }

    operator const QRegExp&() const noexcept { return m_regex; }
};
#endif

// ----------------------------------------------------------------------------
// QTextCodec Compatibility
// ----------------------------------------------------------------------------
// Qt5: QTextCodec (available)
// Qt6: QTextCodec removed, use QStringConverter instead
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: QTextCodec is removed, use QStringConverter
#include <QByteArray>
#include <QStringConverter>

// Forward declaration for compatibility wrapper
class QtCompatTextCodec;

// Compatibility wrapper class for QTextCodec API
class QtCompatTextCodec {
private:
    QStringConverter::Encoding m_encoding;

public:
    QtCompatTextCodec(
            QStringConverter::Encoding encoding = QStringConverter::System)
        : m_encoding(encoding) {}

    // Convert from Unicode (QString) to encoding (QByteArray)
    QByteArray fromUnicode(const QString& str) {
        QStringEncoder encoder(m_encoding);
        if (!encoder.isValid()) {
            // Fallback to UTF-8 if encoding is not available
            encoder = QStringEncoder(QStringConverter::Utf8);
        }
        auto result = encoder.encode(str);
        if (encoder.hasError()) {
            // If encoding fails, fallback to UTF-8
            QStringEncoder utf8Encoder(QStringConverter::Utf8);
            return utf8Encoder.encode(str);
        }
        return result;
    }

    // Convert from encoding (const char*) to Unicode (QString)
    QString toUnicode(const char* chars, int len = -1) {
        QStringDecoder decoder(m_encoding);
        if (!decoder.isValid()) {
            // Fallback to UTF-8 if encoding is not available
            decoder = QStringDecoder(QStringConverter::Utf8);
        }
        QByteArray ba;
        if (len < 0) {
            ba = QByteArray(chars);
        } else {
            ba = QByteArray(chars, len);
        }
        auto result = decoder.decode(ba);
        if (decoder.hasError()) {
            // If decoding fails, try UTF-8
            QStringDecoder utf8Decoder(QStringConverter::Utf8);
            return utf8Decoder.decode(ba);
        }
        return result;
    }

    // Convert from encoding (QByteArray) to Unicode (QString)
    QString toUnicode(const QByteArray& ba) {
        QStringDecoder decoder(m_encoding);
        if (!decoder.isValid()) {
            // Fallback to UTF-8 if encoding is not available
            decoder = QStringDecoder(QStringConverter::Utf8);
        }
        auto result = decoder.decode(ba);
        if (decoder.hasError()) {
            // If decoding fails, try UTF-8
            QStringDecoder utf8Decoder(QStringConverter::Utf8);
            return utf8Decoder.decode(ba);
        }
        return result;
    }
};

// Compatibility function to get codec for locale (similar to
// QTextCodec::codecForLocale())
inline QtCompatTextCodec* qtCompatCodecForLocale() {
    static QtCompatTextCodec codec(QStringConverter::System);
    return &codec;
}

// Compatibility function to get codec by name (similar to
// QTextCodec::codecForName())
inline QtCompatTextCodec* qtCompatCodecForName(const char* name) {
    if (!name) {
        return qtCompatCodecForLocale();
    }

    QString nameStr = QString::fromLatin1(name).toLower();

    // Handle common encoding names
    if (nameStr == "utf-8" || nameStr == "utf8") {
        static QtCompatTextCodec utf8Codec(QStringConverter::Utf8);
        return &utf8Codec;
    } else if (nameStr == "utf-16" || nameStr == "utf16") {
        static QtCompatTextCodec utf16Codec(QStringConverter::Utf16);
        return &utf16Codec;
    } else if (nameStr == "utf-16le" || nameStr == "utf16le") {
        static QtCompatTextCodec utf16leCodec(QStringConverter::Utf16LE);
        return &utf16leCodec;
    } else if (nameStr == "utf-16be" || nameStr == "utf16be") {
        static QtCompatTextCodec utf16beCodec(QStringConverter::Utf16BE);
        return &utf16beCodec;
    } else if (nameStr == "latin1" || nameStr == "iso-8859-1") {
        static QtCompatTextCodec latin1Codec(QStringConverter::Latin1);
        return &latin1Codec;
    }

    // For other encodings, try to use system encoding or fallback to UTF-8
    // In Qt6, QStringConverter doesn't support all encodings that QTextCodec
    // did So we fallback to system encoding or UTF-8
    return qtCompatCodecForLocale();
}

// Type alias for compatibility
using QtCompatQTextCodec = QtCompatTextCodec;

#else
// Qt5: Use QTextCodec directly
#include <QTextCodec>

// Type alias for compatibility
using QtCompatQTextCodec = QTextCodec;

// Compatibility functions
inline QTextCodec* qtCompatCodecForLocale() {
    return QTextCodec::codecForLocale();
}

inline QTextCodec* qtCompatCodecForName(const char* name) {
    return QTextCodec::codecForName(name);
}
#endif

// ----------------------------------------------------------------------------
// QTextStream::endl Compatibility
// ----------------------------------------------------------------------------
// Qt5: endl is a global function
// Qt6: Qt::endl (moved to Qt namespace)
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use Qt::endl
namespace QtCompat {
inline QTextStream& endl(QTextStream& stream) { return Qt::endl(stream); }
}  // namespace QtCompat

#define QTCOMPAT_ENDL Qt::endl
#else
// Qt5: Implement endl directly to avoid conflicts with std::endl
// Qt's endl writes a newline and flushes the stream
namespace QtCompat {
inline QTextStream& endl(QTextStream& stream) {
    stream << QLatin1Char('\n');
    stream.flush();
    return stream;
}
}  // namespace QtCompat

// Use QtCompat::endl for the macro to avoid std::endl conflicts
#define QTCOMPAT_ENDL QtCompat::endl
#endif

// ----------------------------------------------------------------------------
// QWheelEvent Compatibility
// ----------------------------------------------------------------------------
// Qt5: QWheelEvent::delta() returns int, QWheelEvent::pos() returns QPoint
// Qt6: QWheelEvent::delta() removed, use angleDelta().y() (returns int),
//      QWheelEvent::position() returns QPointF
// Compatibility functions return double and QPointF for consistency.
// Note: QWheelEvent is in QtWidgets module. Files using these functions must
// link QtWidgets.
// ----------------------------------------------------------------------------

#include <QWheelEvent>

// Compatibility function for QWheelEvent::delta() - returns double
inline double qtCompatWheelEventDelta(const QWheelEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return static_cast<double>(event->angleDelta().y());
#else
    return static_cast<double>(event->delta());
#endif
}

// Compatibility function for QWheelEvent::pos() - returns QPointF
inline QPointF qtCompatWheelEventPos(const QWheelEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return event->position();
#else
    return QPointF(event->pos());
#endif
}

// ----------------------------------------------------------------------------
// QMouseEvent Compatibility
// ----------------------------------------------------------------------------
// Qt5: QMouseEvent::pos() returns QPoint, globalPos() returns QPoint
// Qt6: QMouseEvent::pos() removed, use position() (returns QPointF),
//      globalPos() removed, use globalPosition() (returns QPointF)
// ----------------------------------------------------------------------------

#include <QMouseEvent>

// Compatibility function for QMouseEvent::pos() - returns QPointF
inline QPointF qtCompatMouseEventPos(const QMouseEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return event->position();
#else
    return QPointF(event->pos());
#endif
}

// Compatibility function for QMouseEvent::globalPos() - returns QPointF
inline QPointF qtCompatMouseEventGlobalPos(const QMouseEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return event->globalPosition();
#else
    return QPointF(event->globalPos());
#endif
}

// Compatibility function for QMouseEvent::pos() - returns QPoint (integer)
inline QPoint qtCompatMouseEventPosInt(const QMouseEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return event->position().toPoint();
#else
    return event->pos();
#endif
}

// Compatibility function for QMouseEvent::globalPos() - returns QPoint
// (integer)
inline QPoint qtCompatMouseEventGlobalPosInt(
        const QMouseEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return event->globalPosition().toPoint();
#else
    return event->globalPos();
#endif
}

// ----------------------------------------------------------------------------
// QDropEvent Compatibility
// ----------------------------------------------------------------------------
// Qt5: QDropEvent::pos() returns QPoint
// Qt6: QDropEvent::pos() removed, use position() (returns QPointF)
// ----------------------------------------------------------------------------

#include <QDropEvent>

// Compatibility function for QDropEvent::pos() - returns QPointF
inline QPointF qtCompatDropEventPos(const QDropEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return event->position();
#else
    return QPointF(event->pos());
#endif
}

// Compatibility function for QDropEvent::pos() - returns QPoint (integer)
inline QPoint qtCompatDropEventPosInt(const QDropEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return event->position().toPoint();
#else
    return event->pos();
#endif
}

// ----------------------------------------------------------------------------
// QMap::insertMulti() / QMultiMap Compatibility
// ----------------------------------------------------------------------------
// Qt5: QMap::insertMulti() allows multiple values per key
// Qt6: QMap::insertMulti() removed, use QMultiMap or insert() which replaces
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: QMap::insertMulti() is removed, use QMultiMap
// For compatibility, we provide a template function that works with QMultiMap
template <typename Key, typename T>
void qtCompatMapInsertMulti(QMap<Key, T>* map, const Key& key, const T& value) {
    // In Qt6, QMap doesn't support insertMulti, so we need to use QMultiMap
    // But if the map is actually a QMultiMap, we can use insert
    // For now, we'll use insert which in Qt6 QMap replaces the value
    // If you need multi-value behavior, use QMultiMap explicitly
    map->insert(key, value);
}

// For QMultiMap, insert works correctly
template <typename Key, typename T>
void qtCompatMapInsertMulti(QMultiMap<Key, T>* map,
                            const Key& key,
                            const T& value) {
    map->insert(key, value);
}

// Compatibility for QMap::unite() - removed in Qt6
template <typename Key, typename T>
void qtCompatMapUnite(QMap<Key, T>* map, const QMap<Key, T>& other) {
    // In Qt6, use insert() which accepts the entire map
    map->insert(other);
}

template <typename Key, typename T>
void qtCompatMapUnite(QMultiMap<Key, T>* map, const QMultiMap<Key, T>& other) {
    // In Qt6, use insert() which accepts the entire map
    map->insert(other);
}
#else
// Qt5: Direct passthrough functions
template <typename Key, typename T>
void qtCompatMapInsertMulti(QMap<Key, T>* map, const Key& key, const T& value) {
    map->insertMulti(key, value);
}

template <typename Key, typename T>
void qtCompatMapInsertMulti(QMultiMap<Key, T>* map,
                            const Key& key,
                            const T& value) {
    map->insert(key, value);
}

template <typename Key, typename T>
void qtCompatMapUnite(QMap<Key, T>* map, const QMap<Key, T>& other) {
    map->unite(other);
}

template <typename Key, typename T>
void qtCompatMapUnite(QMultiMap<Key, T>* map, const QMultiMap<Key, T>& other) {
    map->unite(other);
}
#endif

// ----------------------------------------------------------------------------
// QVariant Type Compatibility
// ----------------------------------------------------------------------------
// Qt5: QVariant::type() returns QVariant::Type enum
// Qt6: QVariant::type() removed, use typeId() which returns QMetaType::Type
// ----------------------------------------------------------------------------

#include <QVariant>

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use typeId() and QMetaType
using QtCompatVariantType = QMetaType::Type;

inline QMetaType::Type qtCompatVariantType(const QVariant& var) noexcept {
    return static_cast<QMetaType::Type>(var.typeId());
}

inline bool qtCompatVariantIsValid(const QVariant& var) noexcept {
    return var.isValid();
}

inline bool qtCompatVariantIsNull(const QVariant& var) noexcept {
    return var.isNull();
}

// Type checking helpers
inline bool qtCompatVariantIsString(const QVariant& var) noexcept {
    return var.typeId() == QMetaType::QString;
}

inline bool qtCompatVariantIsInt(const QVariant& var) noexcept {
    return var.typeId() == QMetaType::Int;
}

inline bool qtCompatVariantIsDouble(const QVariant& var) noexcept {
    return var.typeId() == QMetaType::Double;
}

inline bool qtCompatVariantIsBool(const QVariant& var) noexcept {
    return var.typeId() == QMetaType::Bool;
}

inline bool qtCompatVariantIsList(const QVariant& var) noexcept {
    return var.typeId() == QMetaType::QVariantList;
}

inline bool qtCompatVariantIsMap(const QVariant& var) noexcept {
    return var.typeId() == QMetaType::QVariantMap;
}

#else
// Qt5: Use type() and QVariant::Type
using QtCompatVariantType = QVariant::Type;

inline QVariant::Type qtCompatVariantType(const QVariant& var) noexcept {
    return var.type();
}

inline bool qtCompatVariantIsValid(const QVariant& var) noexcept {
    return var.isValid();
}

inline bool qtCompatVariantIsNull(const QVariant& var) noexcept {
    return var.isNull();
}

// Type checking helpers
inline bool qtCompatVariantIsString(const QVariant& var) noexcept {
    return var.type() == QVariant::String;
}

inline bool qtCompatVariantIsInt(const QVariant& var) noexcept {
    return var.type() == QVariant::Int;
}

inline bool qtCompatVariantIsDouble(const QVariant& var) noexcept {
    return var.type() == QVariant::Double;
}

inline bool qtCompatVariantIsBool(const QVariant& var) noexcept {
    return var.type() == QVariant::Bool;
}

inline bool qtCompatVariantIsList(const QVariant& var) noexcept {
    return var.type() == QVariant::List;
}

inline bool qtCompatVariantIsMap(const QVariant& var) noexcept {
    return var.type() == QVariant::Map;
}
#endif

// ----------------------------------------------------------------------------
// QPlainTextEdit::setTabStopWidth() / setTabStopDistance() Compatibility
// ----------------------------------------------------------------------------
// Qt5: QPlainTextEdit::setTabStopWidth(int width) (deprecated in Qt5.10+)
// Qt6: QPlainTextEdit::setTabStopDistance(qreal distance) (setTabStopWidth
// removed)
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// Qt6: Use setTabStopDistance() which accepts qreal
#include <QPlainTextEdit>
inline void qtCompatSetTabStopWidth(QPlainTextEdit* edit, int width) {
    edit->setTabStopDistance(static_cast<qreal>(width));
}
#else
// Qt5: Use setTabStopWidth() if available, otherwise use setTabStopDistance()
#include <QPlainTextEdit>
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
// Qt5.10+: setTabStopWidth is deprecated, prefer setTabStopDistance
inline void qtCompatSetTabStopWidth(QPlainTextEdit* edit, int width) {
    edit->setTabStopDistance(static_cast<qreal>(width));
}
#else
// Qt5.0-5.9: Use setTabStopWidth
inline void qtCompatSetTabStopWidth(QPlainTextEdit* edit, int width) {
    edit->setTabStopWidth(width);
}
#endif
#endif

// ----------------------------------------------------------------------------
// QSet / QVector Iterator Range Constructor Compatibility
// ----------------------------------------------------------------------------
// Qt5.0-5.14: QSet<T>(begin, end) and QVector<T>(begin, end) constructors not
// supported Qt5.15+: Both support iterator range constructors Qt6: Both support
// iterator range constructors
// ----------------------------------------------------------------------------

// Helper to create QSet from QVector (Qt5/Qt6 compatible)
template <typename T>
QSet<T> qtCompatQSetFromVector(const QVector<T>& vec) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    // Qt5.15+ and Qt6: Use iterator range constructor
    return QSet<T>(vec.begin(), vec.end());
#else
    // Qt5.0-5.14: Manual insertion (iterator range constructor not available)
    QSet<T> result;
    result.reserve(vec.size());
    for (const T& item : vec) {
        result.insert(item);
    }
    return result;
#endif
}

// Helper to create QVector from QSet (Qt5/Qt6 compatible)
template <typename T>
QVector<T> qtCompatQVectorFromSet(const QSet<T>& set) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    // Qt5.15+ and Qt6: Use iterator range constructor
    return QVector<T>(set.begin(), set.end());
#else
    // Qt5.0-5.14: Manual conversion (iterator range constructor not available)
    QVector<T> result;
    result.reserve(set.size());
    for (const T& item : set) {
        result.append(item);
    }
    return result;
#endif
}

// Backward compatibility aliases (for code already using these names)
template <typename T>
inline QSet<T> qSetFromVector(const QVector<T>& vec) {
    return qtCompatQSetFromVector(vec);
}

template <typename T>
inline QVector<T> qVectorFromSet(const QSet<T>& set) {
    return qtCompatQVectorFromSet(set);
}
