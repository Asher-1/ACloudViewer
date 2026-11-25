// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Qt5/Qt6 Compatibility Layer
//
// Supported Qt range: every release from Qt 5.12 through Qt 6 (all 6.x
// minors). Version-dependent APIs must resolve through QT_VERSION guards
// here, never through a customized Qt installation.
//
// This header provides compatibility wrappers for APIs that differ between Qt5
// and Qt6. Include this header instead of directly using Qt headers for
// compatibility-sensitive APIs.
//
// SUPPORTED COMPATIBILITY FEATURES:
//
// 1. Regular Expressions:
//    - QRegularExpression on both Qt majors; no legacy Qt module
//    - Functions: qtCompatRegExp(), qtCompatSplit(), qtCompatSplitRegex(),
//                 qtCompatReplace()
//    - Native match objects own captures; anchoredPattern() makes exact matches
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
//    - QTextCodec (Qt5 Core) / QStringConverter and native codecs (Qt6)
//    - Type alias: QtCompatQTextCodec
//    - Functions: qtCompatCodecForLocale(), qtCompatCodecForName()
//
// 6. Text Stream:
//    - Function: qtCompatSetTextStreamUtf8() (codec / encoding)
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
// 14. Atomic Integer Relaxed Access:
//    - QAtomicInteger::loadRelaxed() / storeRelaxed() (Qt5.14+, Qt6)
//    - Qt5.0-5.13: load() / store() only
//    - Functions: qtCompatLoadRelaxed(), qtCompatStoreRelaxed()
//
// 15. QImage BGR888 Format:
//    - QImage::Format_BGR888 (Qt5.14+, Qt6) / not available on Qt5 < 5.14
//    - Function: qtCompatQImageFormatBgr888()
//
// 16. QFont Family Fallback:
//    - QFont::setFamilies(QStringList) (Qt5.13+, Qt6) / setFamily() only on
//      Qt5.0-5.12
//    - Function: qtCompatSetFontFamilies(QFont&, const QStringList&)
//      (batch list on Qt5.13+/Qt6; first candidate + per-glyph fallback
//      chain on older Qt5)
//
// 17. QDataStream Serialization Version:
//    - Fixed Qt_5_12 wire format for data shared by Qt5 and Qt6 builds.
//    - Function: qtCompatDataStreamVersion()
//      (SDK upgrades must not silently change a persistent archive format)
//
// 18. Markdown and Text Block Headings:
//    - QTextDocument Markdown / QTextBlockFormat headings (Qt5.14+, Qt6)
//    - Functions: qtCompatSetMarkdown(), qtCompatToMarkdown(),
//                 qtCompatHeadingLevel(), qtCompatSetHeadingLevel()
//
// 19. Optional Image Formats and Palette Roles:
//    - Functions: qtCompatQImageFormatGrayscale16() (Qt5.13+),
//                 qtCompatSetPlaceholderTextColor() (Qt5.12+)
//    - Older Qt returns Format_Invalid / preserves all palette roles.
//
// 20. UUID Formatting:
//    - QUuid formatting enums/overloads are unavailable before Qt5.11.
//    - Functions: qtCompatUuidWithoutBraces(), qtCompatUuidId128()
//
// 21. Exclusive File Creation:
//    - Function: qtCompatOpenNewFile() (QIODevice::NewOnly, Qt5.11+)
//    - Older Qt uses atomic native O_EXCL creation, never exists()+truncate.
//
// 22. Font Database Families:
//    - QFontDatabase::families() is an instance method on Qt5 and a static
//      member on Qt6 (the constructor was removed)
//    - Function: qtCompatFontFamilies()
//
// 23. Tab Bar Tab Visibility:
//    - QTabBar::isTabVisible() (Qt5.15+/Qt6); older Qt5 cannot hide tabs
//    - Function: qtCompatTabBarIsTabVisible()
//
// 24. Push Button Bevel Sub-Element:
//    - QStyle::SE_PushButtonBevel (Qt6.7+); older Qt keeps a private value
//      above SE_CustomBase for styles that handle it themselves
//    - Function: qtCompatPushButtonBevelSubElement()
//
// 25. Enter Event Argument Type:
//    - QWidget::enterEvent(QEvent*) (Qt5) / enterEvent(QEnterEvent*) (Qt6)
//    - Type alias: QtCompatEnterEvent (use in override signatures)
//
// 26. Button Group Integer Clicked Signal:
//    - QButtonGroup::buttonClicked(int) (Qt5.0-5.14) / idClicked(int)
//      (Qt5.15+, Qt6; the int overload is removed in Qt6)
//    - Function: qtCompatButtonGroupClicked() (member pointer for connect())
//
// 27. Markdown Bold HTML Export Weight:
//    - QTextDocument::toHtml() serializes markdown bold as font-weight:600
//      on Qt5 and font-weight:700 on Qt6 (QFont weight scale change)
//    - Function: qtCompatMarkdownBoldWeight()
//
// 28. File Dialog Select-and-Accept Timing:
//    - QFileDialog::selectFile() is synchronous on Qt5 but only takes effect
//      once the directory model loaded on Qt6; until then selectedFiles()
//      reports the stale directory and accept() validates the selection
//    - Function: qtCompatFileDialogSelectAndAccept()
//
// Module boundary: this header only includes QtCore/QtWidgets headers, which
// every consumer already links. Helpers needing another Qt module (e.g. Svg)
// belong to the consuming library's own compatibility surface, not here.
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
//   New: static const auto filter = qtCompatRegExp("pattern");
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

#include <QAtomicInteger>
#include <QMap>
#include <QMultiMap>
#include <QPoint>
#include <QPointF>
#include <QRegularExpression>
#include <QSet>
#include <QString>
#include <QStringList>
#include <QTextStream>
#include <QVector>
#include <QtGlobal>
#include <iterator>
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QStringView>
#include <Qt>
#ifdef Q_OS_WIN
#include <QtCore/qt_windows.h>
#include <mlang.h>
#else
#include <iconv.h>

#include <cerrno>
#include <cstring>
#endif
#else
#include <QChar>
#include <QStringRef>
#include <QVector>
#endif

#include <QFile>
#include <QUuid>
#if QT_VERSION < QT_VERSION_CHECK(5, 11, 0)
#include <fcntl.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#if defined(Q_OS_WIN)
#include <io.h>
#else
#include <unistd.h>
#endif
#endif

inline bool qtCompatOpenNewFile(QFile& file, QString* error = nullptr) {
    if (error) error->clear();
    if (file.isOpen()) {
        if (error) *error = QStringLiteral("File is already open");
        return false;
    }
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
    const bool opened = file.open(QIODevice::WriteOnly | QIODevice::NewOnly);
    if (!opened && error) *error = file.errorString();
    return opened;
#else
    // Creation and the non-overwrite check must be one atomic operation.
#if defined(Q_OS_WIN)
    const int fd =
            ::_wopen(reinterpret_cast<const wchar_t*>(file.fileName().utf16()),
                     _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY | _O_NOINHERIT,
                     _S_IREAD | _S_IWRITE);
#else
    const int fd = ::open(QFile::encodeName(file.fileName()).constData(),
                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
#endif
    if (fd < 0) {
        if (error) *error = QString::fromLocal8Bit(std::strerror(errno));
        return false;
    }
    if (file.open(fd, QIODevice::WriteOnly, QFileDevice::AutoCloseHandle))
        return true;
    if (error) *error = file.errorString();
#if defined(Q_OS_WIN)
    ::_close(fd);
#else
    ::close(fd);
#endif
    return false;
#endif
}

// Default toString() is the same braced canonical UUID on all supported Qt.
inline QString qtCompatUuidWithoutBraces(const QUuid& uuid) {
    return uuid.toString().mid(1, 36);
}

inline QString qtCompatUuidId128(const QUuid& uuid) {
    return qtCompatUuidWithoutBraces(uuid).remove(QLatin1Char('-'));
}

// ----------------------------------------------------------------------------
// Portable regular-expression construction
// ----------------------------------------------------------------------------
// The applications use Unicode words and allow dot to span newlines. State,
// captures and exact matches belong to QRegularExpressionMatch/anchoredPattern,
// not a mutable emulation of the removed QRegExp API. Use \z for a strict end
// anchor; QRegularExpression's $ also accepts a position before a final
// newline.
// ----------------------------------------------------------------------------

inline QRegularExpression qtCompatRegExp(
        const QString& pattern,
        QRegularExpression::PatternOptions options =
                QRegularExpression::NoPatternOption) {
    return QRegularExpression(
            pattern, options | QRegularExpression::UseUnicodePropertiesOption |
                             QRegularExpression::DotMatchesEverythingOption);
}

inline QStringList qtCompatSplit(
        const QString& str,
        const QRegularExpression& regex,
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
        Qt::SplitBehavior behavior = Qt::KeepEmptyParts) {
#else
        QString::SplitBehavior behavior = QString::KeepEmptyParts) {
#endif
    return str.split(regex, behavior);
}

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
    return QStringView(str).split(sep);
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
// Qt 5.15 deprecated QString's flags; Qt 5.12 still requires them.
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
namespace QtCompat {
constexpr Qt::SplitBehavior SkipEmptyParts = Qt::SkipEmptyParts;
constexpr Qt::SplitBehavior KeepEmptyParts = Qt::KeepEmptyParts;
}  // namespace QtCompat
#else
namespace QtCompat {
constexpr QString::SplitBehavior SkipEmptyParts = QString::SkipEmptyParts;
constexpr QString::SplitBehavior KeepEmptyParts = QString::KeepEmptyParts;
}  // namespace QtCompat
#endif

inline QStringList qtCompatSplitRegex(
        const QString& str,
        const QString& pattern,
        decltype(QtCompat::KeepEmptyParts) behavior =
                QtCompat::KeepEmptyParts) {
    return qtCompatSplit(str, qtCompatRegExp(pattern), behavior);
}

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
// QString::replace() with regular expressions
// ----------------------------------------------------------------------------

inline QString& qtCompatReplace(QString& str,
                                const QRegularExpression& regex,
                                const QString& after) {
    return str.replace(regex, after);
}

inline QString qtCompatReplace(const QString& str,
                               const QRegularExpression& regex,
                               const QString& after) {
    QString result = str;
    qtCompatReplace(result, regex, after);
    return result;
}

inline QString& qtCompatReplace(QString& str,
                                const QString& pattern,
                                const QString& after) {
    return qtCompatReplace(str, qtCompatRegExp(pattern), after);
}

inline QString qtCompatReplace(const QString& str,
                               const QString& pattern,
                               const QString& after) {
    return qtCompatReplace(str, qtCompatRegExp(pattern), after);
}

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
// QTextCodec Compatibility
// ----------------------------------------------------------------------------
// Unicode wire formats retain Qt5 one-shot behavior. Legacy codecs use the
// operating system; unavailable/unknown names return nullptr.
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QMutex>
#include <QMutexLocker>
#include <QStringConverter>
#include <limits>
#include <map>
#else
#include <QTextCodec>
#endif

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// QtCore handles Unicode/Latin1/System. Legacy archive names use the platform's
// charset service, not an optional Qt module or a silent UTF8 fallback. Each
// conversion owns its state; cached codecs are immutable and shared safely.
class QtCompatTextCodec {
public:
    static QtCompatTextCodec* codecForLocale() {
        static QtCompatTextCodec codec("System", QStringConverter::System);
        return &codec;
    }

    static QtCompatTextCodec* codecForName(const char* name) {
        if (!name || !*name) return nullptr;
        QByteArray key(name);
        key = key.toLower();
        auto encoding = QStringConverter::encodingForName(name);
        if (encoding) key = QStringConverter::nameForEncoding(*encoding);
        // Qt's Shift-JIS codec includes the Windows-31J extension repertoire.
        QByteArray compact = key;
        compact.replace("-", "").replace("_", "");
        if (compact == "shiftjis" || compact == "sjis" || compact == "cp932" ||
            compact == "mskanji" || compact == "csshiftjis") {
            key = "CP932";
        } else if (compact == "windows1252" || compact == "cp1252") {
            key = "CP1252";
        } else if (compact == "iso2022jp" || compact == "csiso2022jp") {
            key = "ISO-2022-JP";
        }
        static QMutex mutex;
        static std::map<QByteArray, QtCompatTextCodec> codecs;
        QMutexLocker lock(&mutex);
        auto found = codecs.find(key);
        if (found != codecs.end()) return &found->second;
        unsigned codePage = 0;
        if (!encoding) {
#ifdef Q_OS_WIN
            codePage = windowsCodePage(key);
            if (!codePage) return nullptr;
#else
            iconv_t encoder = iconv_open(key.constData(), utf16Name());
            if (encoder == reinterpret_cast<iconv_t>(-1)) return nullptr;
            iconv_close(encoder);
            iconv_t decoder = iconv_open(utf16Name(), key.constData());
            if (decoder == reinterpret_cast<iconv_t>(-1)) return nullptr;
            iconv_close(decoder);
#endif
        }
        return &codecs.emplace(key, QtCompatTextCodec(key, encoding, codePage))
                        .first->second;
    }

    QByteArray name() const { return m_name; }

    QByteArray fromUnicode(const QString& text) const {
        if (!m_encoding) {
#ifdef Q_OS_WIN
            QString input = normalizeJapanese(text, true);
            // Qt5 leaves a terminal high surrogate pending and substitutes
            // other malformed code units without changing valid pairs.
            for (qsizetype i = 0; i < input.size(); ++i) {
                if (input.at(i).isHighSurrogate()) {
                    if (i + 1 == input.size()) {
                        input.chop(1);
                        break;
                    }
                    if (input.at(i + 1).isLowSurrogate()) {
                        ++i;
                        continue;
                    }
                    input[i] = QChar(u'?');
                } else if (input.at(i).isLowSurrogate()) {
                    input[i] = QChar(u'?');
                }
            }
            if (input.isEmpty()) return {};
            const DWORD flags =
                    m_codePage == 42 ||
                                    (m_codePage >= 50220 &&
                                     m_codePage <= 50229) ||
                                    m_codePage == 52936 ||
                                    m_codePage == 54936 ||
                                    m_codePage >= 65000 ||
                                    (m_codePage >= 57002 && m_codePage <= 57011)
                            ? 0
                            : WC_NO_BEST_FIT_CHARS;
            const auto* data = reinterpret_cast<const wchar_t*>(input.utf16());
            if (input.size() > std::numeric_limits<int>::max()) return {};
            const int length = static_cast<int>(input.size());
            const int size =
                    WideCharToMultiByte(m_codePage, flags, data, length,
                                        nullptr, 0, nullptr, nullptr);
            QByteArray result(size, Qt::Uninitialized);
            if (size)
                WideCharToMultiByte(m_codePage, flags, data, length,
                                    result.data(), size, nullptr, nullptr);
            return result;
#else
            const QString input = normalizeJapanese(text, true);
            return convert(reinterpret_cast<const char*>(input.utf16()),
                           static_cast<size_t>(input.size()) * sizeof(char16_t),
                           true);
#endif
        }
        if (*m_encoding == QStringConverter::Latin1) return text.toLatin1();
        const bool wide = *m_encoding >= QStringConverter::Utf16 &&
                          *m_encoding <= QStringConverter::Utf32BE;
        QStringEncoder encoder(*m_encoding,
                               wide ? QStringConverter::Flag::WriteBom
                                    : QStringConverter::Flag::Default);
        return encoder(text);
    }

    QString toUnicode(const QByteArray& data) const {
        if (!m_encoding) {
#ifdef Q_OS_WIN
            if (data.isEmpty()) return {};
            if (m_codePage == 932) {
                QString result;
                for (qsizetype i = 0; i < data.size();) {
                    const auto byte = static_cast<unsigned char>(data.at(i));
                    int count = IsDBCSLeadByteEx(932, byte) ? 2 : 1;
                    if (count == 2) {
                        if (i + 1 == data.size()) break;
                        const auto trail =
                                static_cast<unsigned char>(data.at(i + 1));
                        if (!((trail >= 0x40 && trail <= 0x7e) ||
                              (trail >= 0x80 && trail <= 0xfc)))
                            count = 1;
                    }
                    wchar_t decoded = 0;
                    const bool valid =
                            byte != 0x80 && byte != 0xa0 && byte < 0xfd &&
                            MultiByteToWideChar(932, MB_ERR_INVALID_CHARS,
                                                data.constData() + i, count,
                                                &decoded, 1) == 1;
                    result.append(valid ? QChar(static_cast<char16_t>(decoded))
                                        : QChar(u'\ufffd'));
                    i += count;
                }
                return result;
            }
            if (data.size() > std::numeric_limits<int>::max()) return {};
            const int length = static_cast<int>(data.size());
            const int size = MultiByteToWideChar(
                    m_codePage, 0, data.constData(), length, nullptr, 0);
            QString result(size, Qt::Uninitialized);
            if (size)
                MultiByteToWideChar(m_codePage, 0, data.constData(), length,
                                    reinterpret_cast<wchar_t*>(result.data()),
                                    size);
            return result;
#else
            const QByteArray decoded = convert(
                    data.constData(), static_cast<size_t>(data.size()), false);
            QString result(decoded.size() / sizeof(char16_t),
                           Qt::Uninitialized);
            if (!decoded.isEmpty()) {
                std::memcpy(result.data(), decoded.constData(),
                            static_cast<size_t>(decoded.size()));
            }
            return normalizeJapanese(result, false);
#endif
        }
        const auto flags =
                *m_encoding == QStringConverter::Utf8 ||
                                *m_encoding == QStringConverter::System
                        ? QStringConverter::Flag::Stateless
                        : QStringConverter::Flag::Default;
        QStringDecoder decoder(*m_encoding, flags);
        return decoder(data);
    }

    QString toUnicode(const char* data, int length = -1) const {
        return toUnicode(length < 0 ? QByteArray(data)
                                    : QByteArray(data, length));
    }

private:
    QtCompatTextCodec(QByteArray name,
                      std::optional<QStringConverter::Encoding> encoding,
                      unsigned codePage = 0)
        : m_name(std::move(name)), m_encoding(encoding), m_codePage(codePage) {}

    // Charset services disagree on six standard JIS Unicode assignments.
    // Keep Qt5 Windows-31J wire mappings, and reject the native service's
    // irreversible best-fit aliases instead of silently renaming an entry.
    QString normalizeJapanese(const QString& text, bool encode) const {
        const bool sjis = m_name == "CP932";
        if (!sjis && m_name != "ISO-2022-JP") return text;
        constexpr std::pair<char16_t, char16_t> mappings[] = {
                {0x301c, 0xff5e}, {0x2016, 0x2225}, {0x2212, 0xff0d},
                {0x00a2, 0xffe0}, {0x00a3, 0xffe1}, {0x00ac, 0xffe2}};
        QString result = text;
        for (qsizetype i = 0; i < result.size(); ++i) {
            const auto value = result.at(i).unicode();
            if (encode && (value == 0x2014 ||
                           (sjis && (value == 0x00a5 || value == 0x203e)))) {
                result[i] = QChar(u'?');
                continue;
            }
            for (const auto& mapping : mappings) {
                if (encode && value == mapping.first) {
                    result[i] = QChar(u'?');
                }
#ifndef Q_OS_WIN
                else if (!sjis &&
                         value == (encode ? mapping.second : mapping.first)) {
                    result[i] = QChar(encode ? mapping.first : mapping.second);
                }
#endif
            }
        }
        return result;
    }

#ifdef Q_OS_WIN
    static unsigned windowsCodePage(const QByteArray& name) {
        if (name.startsWith("CP")) {
            bool valid = false;
            const unsigned page = name.mid(2).toUInt(&valid);
            if (valid && IsValidCodePage(page)) return page;
        }
        const HRESULT initialized =
                CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(initialized) && initialized != RPC_E_CHANGED_MODE) return 0;
        IMultiLanguage2* service = nullptr;
        unsigned result = 0;
        if (SUCCEEDED(CoCreateInstance(
                    CLSID_CMultiLanguage, nullptr, CLSCTX_INPROC_SERVER,
                    IID_IMultiLanguage2, reinterpret_cast<void**>(&service)))) {
            const QString charset = QString::fromLatin1(name);
            BSTR value = SysAllocString(
                    reinterpret_cast<const wchar_t*>(charset.utf16()));
            MIMECSETINFO info{};
            if (value && SUCCEEDED(service->GetCharsetInfo(value, &info)) &&
                IsValidCodePage(info.uiInternetEncoding)) {
                result = info.uiInternetEncoding;
            }
            SysFreeString(value);
            service->Release();
        }
        if (SUCCEEDED(initialized)) CoUninitialize();
        return result;
    }
#else
    static const char* utf16Name() {
#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
        return "UTF-16LE";
#else
        return "UTF-16BE";
#endif
    }

    QByteArray convert(const char* data, size_t length, bool encode) const {
        iconv_t descriptor =
                iconv_open(encode ? m_name.constData() : utf16Name(),
                           encode ? utf16Name() : m_name.constData());
        if (descriptor == reinterpret_cast<iconv_t>(-1)) return {};
        QByteArray result;
        char* input = const_cast<char*>(data);
        while (length) {
            char buffer[1024];
            char* output = buffer;
            size_t available = sizeof(buffer);
            const size_t status =
                    iconv(descriptor, &input, &length, &output, &available);
            result.append(buffer,
                          static_cast<qsizetype>(sizeof(buffer) - available));
            if (status != static_cast<size_t>(-1) || errno == E2BIG) continue;
            // Like Qt5's one-shot legacy codecs, retain no incomplete tail.
            if (errno == EINVAL) break;
            if (errno != EILSEQ) {
                result.clear();
                break;
            }
            if (encode) {
                char16_t first;
                std::memcpy(&first, input, sizeof(first));
                if (m_name == "CP1252" &&
                    (first == 0x81 || first == 0x8d || first == 0x8f ||
                     first == 0x90 || first == 0x9d)) {
                    result.append(static_cast<char>(first));
                    input += sizeof(first);
                    length -= sizeof(first);
                    continue;
                }
                size_t skip = sizeof(first);
                if (QChar::isHighSurrogate(first) &&
                    length >= 2 * sizeof(first)) {
                    char16_t second;
                    std::memcpy(&second, input + sizeof(first), sizeof(second));
                    if (QChar::isLowSurrogate(second)) skip *= 2;
                }
                input += skip;
                length -= skip;
                // Encode the substitution through the same stateful descriptor.
                char16_t question = u'?';
                char* replacement = reinterpret_cast<char*>(&question);
                size_t remaining = sizeof(question);
                output = buffer;
                available = sizeof(buffer);
                iconv(descriptor, &replacement, &remaining, &output,
                      &available);
                result.append(buffer, static_cast<qsizetype>(sizeof(buffer) -
                                                             available));
            } else {
                const auto byte = static_cast<unsigned char>(*input++);
                --length;
                if (m_name == "CP932" && length &&
                    ((byte >= 0x81 && byte <= 0x9f) ||
                     (byte >= 0xe0 && byte <= 0xfc))) {
                    const auto trail = static_cast<unsigned char>(*input);
                    if ((trail >= 0x40 && trail <= 0x7e) ||
                        (trail >= 0x80 && trail <= 0xfc)) {
                        ++input;
                        --length;
                    }
                }
                // Qt5 preserves the five undefined Windows-1252 C1 controls.
                const char16_t replacement =
                        m_name == "CP1252" && (byte == 0x81 || byte == 0x8d ||
                                               byte == 0x8f || byte == 0x90 ||
                                               byte == 0x9d)
                                ? byte
                                : 0xfffd;
                result.append(reinterpret_cast<const char*>(&replacement),
                              sizeof(replacement));
            }
        }
        // Do not emit an extra terminal shift sequence: QTextCodec's one-shot
        // ISO-2022 encoders leave that state pending as well.
        iconv_close(descriptor);
        return result;
    }
#endif
    const QByteArray m_name;
    const std::optional<QStringConverter::Encoding> m_encoding;
    const unsigned m_codePage;
};

using QtCompatQTextCodec = QtCompatTextCodec;
#else
using QtCompatQTextCodec = QTextCodec;
#endif

inline QtCompatQTextCodec* qtCompatCodecForLocale() {
    return QtCompatQTextCodec::codecForLocale();
}

inline QtCompatQTextCodec* qtCompatCodecForName(const char* name) {
    return QtCompatQTextCodec::codecForName(name);
}

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

// Keep label imports and exports independent of the process locale.
inline void qtCompatSetTextStreamUtf8(QTextStream& stream) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    stream.setEncoding(QStringConverter::Utf8);
#else
    stream.setCodec("UTF-8");
#endif
}

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
    return static_cast<double>(event->angleDelta().y());
}

// Compatibility function for QWheelEvent::pos() - returns QPointF
inline QPointF qtCompatWheelEventPos(const QWheelEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    return event->position();
#else
    return event->posF();
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
    // Preserve subpixel coordinates until the consumer requests integer pixels.
    return event->localPos();
#endif
}

// Compatibility function for QMouseEvent::globalPos() - returns QPointF
inline QPointF qtCompatMouseEventGlobalPos(const QMouseEvent* event) noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return event->globalPosition();
#else
    return event->screenPos();
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

/** True for every signed/unsigned integer variant width. JSON numbers
 *  desynchronize across Qt versions: QJsonValue::toVariant() returns Double
 *  on Qt5 but LongLong for integral values on Qt6, so integer-typed RPC
 *  parameters must be accepted through this helper instead of an exact
 *  Int/Double comparison (QtCompat owns the divergence, item 11). */
inline bool qtCompatVariantIsIntegral(const QVariant& var) noexcept {
    const QMetaType::Type id = static_cast<QMetaType::Type>(var.typeId());
    return id == QMetaType::Int || id == QMetaType::UInt ||
           id == QMetaType::LongLong || id == QMetaType::ULongLong ||
           id == QMetaType::Short || id == QMetaType::UShort ||
           id == QMetaType::Char || id == QMetaType::UChar ||
           id == QMetaType::SChar || id == QMetaType::Long ||
           id == QMetaType::ULong;
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

/** True for every signed/unsigned integer variant width (see the Qt6
 *  overload above): Qt5 keeps JSON integers as Double, but explicit
 *  LongLong/ULongLong call sites must stay accepted on both branches.
 *  QVariant::Type has no Long/ULong members on Qt5, so those widths do
 *  not exist as distinct variant types here. */
inline bool qtCompatVariantIsIntegral(const QVariant& var) noexcept {
    const QVariant::Type type = var.type();
    return type == QVariant::Int || type == QVariant::UInt ||
           type == QVariant::LongLong || type == QVariant::ULongLong;
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
// QWidget enter-event signature and QButtonGroup integer signals
// ----------------------------------------------------------------------------
#include <QButtonGroup>
#include <QEvent>

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QEnterEvent>
using QtCompatEnterEvent = QEnterEvent;
#else
using QtCompatEnterEvent = QEvent;
#endif

inline auto qtCompatButtonGroupIdClicked() {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    return &QButtonGroup::idClicked;
#else
    return QOverload<int>::of(&QButtonGroup::buttonClicked);
#endif
}

// ----------------------------------------------------------------------------
// Container/string sizes at int-based Qt widget and application boundaries
// ----------------------------------------------------------------------------
#include <limits>

// Qt 5 returns int while Qt 6 returns qsizetype. Keep int-based callers
// source-compatible without wrapping oversized lengths into negative values.
// Do not use this for traversal of containers larger than INT_MAX.
template <typename Container>
inline int qtCompatSizeInt(const Container& container) noexcept {
    const auto size = container.size();
    return size > (std::numeric_limits<int>::max)()
                   ? (std::numeric_limits<int>::max)()
                   : static_cast<int>(size);
}

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

// Helper to create QSet from a QList/QStringList (Qt5/Qt6 compatible).
// Qt5.0-5.14 keeps QList and QVector as distinct classes, so the QVector
// overload above does not accept a QStringList; this overload covers those
// call sites with the same version split.
template <typename T>
QSet<T> qtCompatQSetFromList(const QList<T>& list) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    // Qt5.15+ and Qt6: Use iterator range constructor (QList IS QVector)
    return QSet<T>(list.begin(), list.end());
#else
    // Qt5.0-5.14: Manual insertion (iterator range constructor not available)
    QSet<T> result;
    result.reserve(list.size());
    for (const T& item : list) {
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

// Helper to create QVector from an arbitrary iterator range [first, last).
// This covers raw pointer ranges (e.g. float* arrays from C APIs) which
// qtCompatQVectorFromSet/qtCompatQSetFromVector do not handle.
template <typename T, typename InputIt>
inline QVector<T> qtCompatQVectorFromRange(InputIt first, InputIt last) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    return QVector<T>(first, last);
#else
    // Qt5.0-5.14: no iterator-range constructor — manual copy loop
    QVector<T> result;
    result.reserve(static_cast<int>(std::distance(first, last)));
    while (first != last) {
        result.append(*first);
        ++first;
    }
    return result;
#endif
}

// ----------------------------------------------------------------------------
// QAtomicInteger Relaxed Load/Store Compatibility
// ----------------------------------------------------------------------------
// Qt5.0-5.13: load() / store()
// Qt5.14+/Qt6: loadRelaxed() / storeRelaxed() (load()/store() deprecated)
// ----------------------------------------------------------------------------

#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
// Qt5.14+/Qt6: relaxed accessors are the canonical API

template <typename T>
inline T qtCompatLoadRelaxed(const QBasicAtomicInteger<T>& atomic) noexcept {
    return atomic.loadRelaxed();
}

template <typename T>
inline void qtCompatStoreRelaxed(QBasicAtomicInteger<T>& atomic,
                                 T value) noexcept {
    atomic.storeRelaxed(value);
}
#else
// Qt5.0-5.13: load()/store() are the relaxed equivalents

template <typename T>
inline T qtCompatLoadRelaxed(const QBasicAtomicInteger<T>& atomic) noexcept {
    return atomic.load();
}

template <typename T>
inline void qtCompatStoreRelaxed(QBasicAtomicInteger<T>& atomic,
                                 T value) noexcept {
    atomic.store(value);
}
#endif

// ----------------------------------------------------------------------------
// QImage::Format_BGR888 Availability
// ----------------------------------------------------------------------------
// Qt5.0-5.13: QImage::Format_BGR888 does not exist (introduced in Qt 5.14)
// Qt5.14+/Qt6: Format_BGR888 available
//
// qtCompatQImageFormatBgr888() returns the BGR888 format value where the
// layout exists. On older Qt5 it returns QImage::Format_Invalid — a value a
// non-null QImage can never report — so `case qtCompatQImageFormatBgr888():`
// labels stay valid, unique, and dead on those versions.
// ----------------------------------------------------------------------------

#include <QImage>

constexpr QImage::Format qtCompatQImageFormatBgr888() noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
    return QImage::Format_BGR888;
#else
    return QImage::Format_Invalid;
#endif
}

// A non-null image never reports Format_Invalid, so older Qt cannot enter
// a consumer's Grayscale16 path. Check for null images before comparing
// formats.
constexpr QImage::Format qtCompatQImageFormatGrayscale16() noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(5, 13, 0)
    return QImage::Format_Grayscale16;
#else
    return QImage::Format_Invalid;
#endif
}

#include <QPalette>

inline void qtCompatSetPlaceholderTextColor(QPalette& palette,
                                            QPalette::ColorGroup group,
                                            const QColor& color) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    palette.setColor(group, QPalette::PlaceholderText, color);
#else
    // No substitute role: changing Text would recolor actual user input.
    Q_UNUSED(palette);
    Q_UNUSED(group);
    Q_UNUSED(color);
#endif
}

// ----------------------------------------------------------------------------
// QFont Family Fallback Compatibility
// ----------------------------------------------------------------------------
// Qt5.0-5.12: QFont only has setFamily(QString) (single family; the font
//   system's per-glyph fallback still resolves other scripts)
// Qt5.13+/Qt6: setFamilies(QStringList) provides an ordered fallback list
// ----------------------------------------------------------------------------

#include <QFont>

inline void qtCompatSetFontFamilies(QFont& font, const QStringList& families) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 13, 0)
    if (!families.isEmpty()) {
        font.setFamilies(families);
        return;
    }
#endif
    if (!families.isEmpty()) {
        font.setFamily(families.first());
    }
}

// ----------------------------------------------------------------------------
// QFontDatabase Families Compatibility
// ----------------------------------------------------------------------------
// Qt5: families() is a member of the (instantiable) QFontDatabase class
// Qt6: the constructor was removed; families() is a static member
// ----------------------------------------------------------------------------

#include <QFontDatabase>

inline QStringList qtCompatFontFamilies() {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return QFontDatabase::families();
#else
    return QFontDatabase().families();
#endif
}

// ----------------------------------------------------------------------------
// QDataStream Serialization Version Compatibility
// ----------------------------------------------------------------------------
// Persistent data needs a fixed wire format, not the SDK's compiled default.
// Qt6 otherwise changes QVariant type IDs and later Qt versions may change
// container lengths. Qt_5_12 is available throughout the supported Qt range
// and preserves the minimum supported Qt5 reader. An archive format upgrade
// must be versioned by its owner; upgrading the Qt SDK is not such an upgrade.
// ----------------------------------------------------------------------------

#include <QDataStream>

inline QDataStream::Version qtCompatDataStreamVersion() noexcept {
    return QDataStream::Qt_5_12;
}

// ----------------------------------------------------------------------------
// QTextDocument Markdown / QTextBlockFormat Heading Compatibility
// ----------------------------------------------------------------------------
// QTextDocument Markdown and heading metadata were added in Qt 5.14.
// Older Qt imports the basic editable formats and fenced code below.
#include <QRegularExpression>
#include <QTextBlock>
#include <QTextBlockFormat>
#include <QTextCursor>
#include <QTextDocument>
#include <QTextDocumentFragment>
#include <QTextFragment>

inline void qtCompatSetMarkdown(QTextDocument& document, const QString& text) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
    document.setMarkdown(text);
#else
    // Preserve the HTML fidelity boundary used by qtCompatToMarkdown().
    if (Qt::mightBeRichText(text)) {
        document.setHtml(text);
        return;
    }
    // Qt < 5.14 has no Markdown importer. Render the basic formats exported
    // below, plus chat code fences; insert text, never unescaped generated
    // HTML.
    document.clear();
    QTextCursor cursor(&document);
    static const QRegularExpression fence("^ {0,3}(`{3,}|~{3,})(.*)$");
    static const QRegularExpression heading("^ {0,3}(#{1,6}) +(.+)$");
    static const QRegularExpression span(
            "(`+)(.+?)\\1|(\\*{1,3})(\\S(?:.*?\\S)?)\\3");
    QString codeFence;
    bool first = true;
    for (QString line : text.split(QLatin1Char('\n'))) {
        const auto marker = fence.match(line);
        if (marker.hasMatch()) {
            if (codeFence.isEmpty()) {
                codeFence = marker.captured(1);
                continue;
            }
            if (marker.captured(1).startsWith(codeFence) &&
                marker.captured(2).trimmed().isEmpty()) {
                codeFence.clear();
                continue;
            }
        }
        QTextBlockFormat block;
        QTextCharFormat format;
        if (!codeFence.isEmpty()) {
            format.setFontFamily(QStringLiteral("monospace"));
            format.setFontFixedPitch(true);
            block.setNonBreakableLines(true);
        } else {
            const auto title = heading.match(line);
            if (title.hasMatch()) {
                block.setProperty(QTextFormat::UserProperty + 1,
                                  title.capturedLength(1));
                format.setFontWeight(QFont::Bold);
                line = title.captured(2);
            }
        }
        if (!first) cursor.insertBlock();
        first = false;
        cursor.setBlockFormat(block);
        if (!codeFence.isEmpty()) {
            cursor.insertText(line, format);
            continue;
        }
        int position = 0;
        auto matches = span.globalMatch(line);
        while (matches.hasNext()) {
            const auto match = matches.next();
            cursor.insertText(
                    line.mid(position, match.capturedStart() - position),
                    format);
            auto emphasis = format;
            if (!match.captured(1).isEmpty()) {
                emphasis.setFontFamily(QStringLiteral("monospace"));
                emphasis.setFontFixedPitch(true);
            } else {
                const int count = match.capturedLength(3);
                if (count >= 2) emphasis.setFontWeight(QFont::Bold);
                if (count != 2) emphasis.setFontItalic(true);
            }
            cursor.insertText(
                    match.captured(match.captured(1).isEmpty() ? 4 : 2),
                    emphasis);
            position = match.capturedEnd();
        }
        cursor.insertText(line.mid(position), format);
    }
#endif
}

inline QString qtCompatToMarkdown(const QTextDocument& document) {
    // Shared export must preserve images, links, underline, strikeout and
    // lists. HTML is Markdown's fidelity boundary; full Markdown import on
    // older Qt remains unimplemented.
    bool preserveHtml = false;
    for (auto block = document.begin(); block.isValid(); block = block.next()) {
        preserveHtml |= block.textList() != nullptr;
        for (auto it = block.begin(); !it.atEnd(); ++it) {
            const auto fragment = it.fragment();
            if (!fragment.isValid()) continue;
            const auto format = fragment.charFormat();
            preserveHtml |= format.fontStrikeOut() || format.fontUnderline() ||
                            format.isImageFormat() || format.isAnchor();
        }
    }
    if (preserveHtml) {
        QTextCursor cursor(document.begin());
        cursor.select(QTextCursor::Document);
        QString result = cursor.selection().toHtml().trimmed();
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
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
    return document.toMarkdown();
#else
    // Linear export of the basic formats handled by the older Qt importer.
    QString result;
    for (auto block = document.begin(); block.isValid(); block = block.next()) {
        if (block != document.begin()) result += QLatin1Char('\n');
        const int heading = block.blockFormat()
                                    .property(QTextFormat::UserProperty + 1)
                                    .toInt();
        if (heading >= 1 && heading <= 6)
            result += QString(heading, QLatin1Char('#')) + QLatin1Char(' ');
        for (auto it = block.begin(); !it.atEnd(); ++it) {
            const QTextFragment fragment = it.fragment();
            if (!fragment.isValid()) continue;
            const auto format = fragment.charFormat();
            QString text = fragment.text();
            if (format.fontWeight() >= QFont::Bold)
                text = QStringLiteral("**") + text + QStringLiteral("**");
            if (format.fontItalic())
                text = QLatin1Char('*') + text + QLatin1Char('*');
            result += text;
        }
    }
    return result;
#endif
}

inline int qtCompatHeadingLevel(const QTextBlockFormat& format) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
    return format.headingLevel();
#else
    return format.property(QTextFormat::UserProperty + 1).toInt();
#endif
}

inline void qtCompatSetHeadingLevel(QTextBlockFormat& format, int level) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
    format.setHeadingLevel(level);
#else
    format.setProperty(QTextFormat::UserProperty + 1, level);
#endif
}

// ----------------------------------------------------------------------------
// QLabel::pixmap() Compatibility
// ----------------------------------------------------------------------------
// Qt5: QLabel::pixmap() returns const QPixmap* (nullptr when no pixmap set)
// Qt6: QLabel::pixmap(Qt::ReturnByValue) returns QPixmap by value
//        (isNull() == true when no pixmap set)
//
// qtCompatLabelPixmap() always returns QPixmap by value on every Qt version.
// ----------------------------------------------------------------------------

#include <QLabel>
#include <QPixmap>

inline QPixmap qtCompatLabelPixmap(const QLabel* label) {
    if (!label) return QPixmap();
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return label->pixmap();
#elif QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    return label->pixmap(Qt::ReturnByValue);
#else
    const QPixmap* pm = label->pixmap();
    return pm ? *pm : QPixmap();
#endif
}

// ----------------------------------------------------------------------------
// QTabBar::isTabVisible() Compatibility
// ----------------------------------------------------------------------------
// Qt5.0-5.14: isTabVisible() does not exist; tabs cannot be hidden, so every
// valid tab is visible.
// Qt5.15+/Qt6: isTabVisible() reports the per-tab visibility flag.
// ----------------------------------------------------------------------------

#include <QTabBar>

inline bool qtCompatTabBarIsTabVisible(const QTabBar* tabBar, int index) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    return tabBar && tabBar->isTabVisible(index);
#else
    Q_UNUSED(tabBar);
    Q_UNUSED(index);
    return true;
#endif
}

// ----------------------------------------------------------------------------
// QStyle::SE_PushButtonBevel Compatibility
// ----------------------------------------------------------------------------
// Qt6.7+ exposes SE_PushButtonBevel. Older Qt has no such enumerator; a style
// that handles the bevel rect in its own subElementRect() override can use a
// private value above SE_CustomBase. Keep that value out of switch labels on
// older Qt (-Wswitch rejects out-of-range case values).
// ----------------------------------------------------------------------------

#include <QStyle>

constexpr QStyle::SubElement qtCompatPushButtonBevelSubElement() noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 7, 0)
    return QStyle::SE_PushButtonBevel;
#else
    return static_cast<QStyle::SubElement>(QStyle::SE_CustomBase + 1);
#endif
}

// ----------------------------------------------------------------------------
// QWidget::enterEvent() Argument Type Compatibility
// ----------------------------------------------------------------------------
// Qt5: enterEvent(QEvent*)
// Qt6: enterEvent(QEnterEvent*)
// Use this alias in override declarations/definitions so one signature
// compiles on every supported Qt version. Only the pointer type is needed
// here; consumers dereferencing the event include its real header.
// ----------------------------------------------------------------------------

#include <QEvent>

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
class QEnterEvent;
using QtCompatEnterEvent = QEnterEvent;
#else
using QtCompatEnterEvent = QEvent;
#endif

// ----------------------------------------------------------------------------
// QButtonGroup Integer Clicked Signal Compatibility
// ----------------------------------------------------------------------------
// Qt5.0-5.14: buttonClicked(int) (overloaded with the QAbstractButton* one)
// Qt5.15+/Qt6: idClicked(int); the int buttonClicked overload is deprecated
//              in 5.15 and removed in Qt6.
// Returns the member pointer for the pointer-to-member connect() overload.
// ----------------------------------------------------------------------------

#include <QButtonGroup>

inline void (QButtonGroup::*qtCompatButtonGroupClicked())(int) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    return &QButtonGroup::idClicked;
#else
    return QOverload<int>::of(&QButtonGroup::buttonClicked);
#endif
}

// ----------------------------------------------------------------------------
// QTextDocument Markdown Bold HTML Export Weight Compatibility
// ----------------------------------------------------------------------------
// Qt5 maps QFont::Bold to CSS font-weight:600 in toHtml(); Qt6 uses the CSS
// weight scale directly (QFont::Bold == 700). Consumers pinning the exported
// HTML must expect the version's value.
// ----------------------------------------------------------------------------

constexpr int qtCompatMarkdownBoldWeight() noexcept {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return 700;
#else
    return 600;
#endif
}

// ----------------------------------------------------------------------------
// QFileDialog Select-and-Accept Timing Compatibility
// ----------------------------------------------------------------------------
// Qt5: selectFile() updates the selection synchronously, so accept() right
// afterwards closes the dialog.
// Qt6: selectFile() only takes effect once the directory model has loaded;
// until then selectedFiles() still reports the previous directory, and
// accept() validates selectedFiles() and refuses to close while it is empty.
// qtCompatFileDialogSelectAndAccept() selects the path and accepts as soon as
// the selection matches the requested path, on every Qt version. The optional
// afterAccept callback runs queued right after the accept (application-scoped,
// so it survives the file dialog's destruction) to drive follow-up modal
// dialogs.
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QTimer>
#include <functional>

inline void qtCompatFileDialogSelectAndAccept(
        QFileDialog* dialog,
        const QString& path,
        const std::function<void()>& afterAccept = std::function<void()>()) {
    if (!dialog) return;
    dialog->selectFile(path);
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    const QString wanted = QDir::cleanPath(path);
    auto* timer = new QTimer(dialog);
    timer->setInterval(20);
    QObject::connect(
            timer, &QTimer::timeout, dialog,
            [dialog, timer, wanted, afterAccept] {
                if (!dialog->isVisible()) {
                    timer->stop();
                    timer->deleteLater();
                    return;
                }
                if (QDir::cleanPath(dialog->selectedFiles().value(0)) != wanted)
                    return;
                timer->stop();
                timer->deleteLater();
                QMetaObject::invokeMethod(dialog, "accept",
                                          Qt::DirectConnection);
                if (afterAccept)
                    QTimer::singleShot(0, QCoreApplication::instance(),
                                       afterAccept);
            });
    timer->start();
#else
    QMetaObject::invokeMethod(dialog, "accept", Qt::DirectConnection);
    if (afterAccept)
        QTimer::singleShot(0, QCoreApplication::instance(), afterAccept);
#endif
}
