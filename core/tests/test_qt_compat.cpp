// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QtCompat.h>
#include <gtest/gtest.h>

TEST(QtCompat, SplitRefKeepsEmptyFields) {
    const QStringList inputs = {QString(), "",   "abc",  ".",    "..",
                                "a.",      ".a", "a..b", "a.b.", "a.b"};
    for (const QString& input : inputs) {
        const QStringList expected =
                input.split(QLatin1Char('.'), QtCompat::KeepEmptyParts);
        const auto actual = qtCompatSplitRefChar(input, QLatin1Char('.'));
        ASSERT_EQ(expected.size(), actual.size()) << input.toStdString();
        for (int i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], qtCompatStringRefToString(actual[i]))
                    << input.toStdString() << " field " << i;
        }
    }
}

TEST(QtCompat, SizeIntSaturatesAtWidgetBoundary) {
    struct Sized {
        qint64 value;
        qint64 size() const noexcept { return value; }
    };
    EXPECT_EQ(0, qtCompatSizeInt(Sized{0}));
    EXPECT_EQ(7, qtCompatSizeInt(Sized{7}));
    const qint64 limit = (std::numeric_limits<int>::max)();
    EXPECT_EQ(limit, qtCompatSizeInt(Sized{limit}));
    EXPECT_EQ(limit, qtCompatSizeInt(Sized{limit + 1}));
}

TEST(QtCompat, UnicodeCodecsPreserveWireFormat) {
    const QString sample = QString::fromUtf8("A\xc3\xa9\xe4\xb8\xad");
    const bool littleEndian = Q_BYTE_ORDER == Q_LITTLE_ENDIAN;
    const char* utf16 = littleEndian ? "fffe4100e9002d4e" : "feff004100e94e2d";
    const char* utf32 = littleEndian ? "fffe000041000000e90000002d4e0000"
                                     : "0000feff00000041000000e900004e2d";
    struct Fixture {
        const char* name;
        const char* encoded;
        const char* empty;
    };
    const Fixture fixtures[] = {
            {"UTF-8", "41c3a9e4b8ad", ""},
            {"UTF-16", utf16, littleEndian ? "fffe" : "feff"},
            {"UTF-16LE", "fffe4100e9002d4e", "fffe"},
            {"UTF-16BE", "feff004100e94e2d", "feff"},
            {"UTF-32", utf32, littleEndian ? "fffe0000" : "0000feff"},
            {"UTF-32LE", "fffe000041000000e90000002d4e0000", "fffe0000"},
            {"UTF-32BE", "0000feff00000041000000e900004e2d", "0000feff"},
    };
    for (const auto& fixture : fixtures) {
        SCOPED_TRACE(fixture.name);
        auto* codec = qtCompatCodecForName(fixture.name);
        ASSERT_NE(nullptr, codec);
        EXPECT_EQ(fixture.encoded,
                  codec->fromUnicode(sample).toHex().toStdString());
        EXPECT_EQ(fixture.empty,
                  codec->fromUnicode(QString()).toHex().toStdString());
        EXPECT_EQ(sample,
                  codec->toUnicode(QByteArray::fromHex(fixture.encoded)));
    }
}

TEST(QtCompat, Latin1LossDoesNotChangeDeclaredEncoding) {
    auto* codec = qtCompatCodecForName("ISO-8859-1");
    ASSERT_NE(nullptr, codec);
    EXPECT_EQ("41e93f",
              codec->fromUnicode(QString::fromUtf8("A\xc3\xa9\xe4\xb8\xad"))
                      .toHex()
                      .toStdString());
    EXPECT_EQ(QString::fromUtf8("A\xc3\xa9?"),
              codec->toUnicode(QByteArray::fromHex("41e93f")));
}

TEST(QtCompat, Utf8CodecCompletesEachIndependentConversion) {
    auto* codec = qtCompatCodecForName("UTF-8");
    ASSERT_NE(nullptr, codec);
    EXPECT_EQ(QString::fromUtf8("\xef\xbf\xbd\xef\xbf\xbd"),
              codec->toUnicode(QByteArray::fromHex("e282")));
    // QTextCodec drops a trailing high surrogate in this one-shot overload.
    EXPECT_TRUE(codec->fromUnicode(QString(1, QChar(0xd800))).isEmpty());
    EXPECT_EQ(QStringLiteral("A"), codec->toUnicode(QByteArray("A")));
    EXPECT_EQ(QByteArray("A"), codec->fromUnicode(QStringLiteral("A")));
}

TEST(QtCompat, CodecCharBufferHonorsExplicitLength) {
    auto* codec = qtCompatCodecForName("UTF-8");
    ASSERT_NE(nullptr, codec);
    const QByteArray bytes("A\0B", 3);
    const QString expected = QString::fromUtf8(bytes.constData(), bytes.size());
    EXPECT_EQ(expected, codec->toUnicode(bytes));
    EXPECT_EQ(expected,
              codec->toUnicode(bytes.constData(), qtCompatSizeInt(bytes)));
    EXPECT_EQ(QStringLiteral("A"), codec->toUnicode(bytes.constData()));
}

TEST(QtCompat, NamedLegacyCodecsDoNotFallBackToUtf8) {
    const QString sample = QString::fromUtf8("A\xc3\xa9\xe4\xb8\xad");
    struct Fixture {
        const char* name;
        const char* bytes;
        const char* decoded;
    };
    const Fixture fixtures[] = {
            {"GB18030", "41a8a6d6d0", "A\xc3\xa9\xe4\xb8\xad"},
            {"Shift-JIS", "413f9286", "A?\xe4\xb8\xad"},
            {"Windows-1252", "41e93f", "A\xc3\xa9?"},
    };
    for (const auto& fixture : fixtures) {
        SCOPED_TRACE(fixture.name);
        auto* codec = qtCompatCodecForName(fixture.name);
        ASSERT_NE(nullptr, codec);
        EXPECT_EQ(QByteArray::fromHex(fixture.bytes),
                  codec->fromUnicode(sample));
        EXPECT_EQ(QString::fromUtf8(fixture.decoded),
                  codec->toUnicode(QByteArray::fromHex(fixture.bytes)));
    }
    EXPECT_EQ(nullptr, qtCompatCodecForName("definitely-not-a-codec"));
    EXPECT_EQ(nullptr, qtCompatCodecForName(""));
}

TEST(QtCompat, RegExpKeepsUnicodeNewlineAndAnchorSemantics) {
    const QString word = QString::fromUtf8("\xc3\xa9\xe4\xb8\xad");
    const auto regex = qtCompatRegExp(QStringLiteral("\\w+"));
    const auto match = regex.match(word);
    EXPECT_EQ(0, match.capturedStart());
    EXPECT_EQ(word, match.captured());
    EXPECT_TRUE(qtCompatRegExp("^a.*b\\z").match("a\nb").hasMatch());
    EXPECT_FALSE(qtCompatRegExp("b\\z").match("b\n").hasMatch());
    EXPECT_EQ(2, qtCompatRegExp("a").match("aba", -1).capturedStart());
    const auto layout = qtCompatRegExp(
            QRegularExpression::anchoredPattern("Layout #(\\d+)"));
    EXPECT_TRUE(layout.match("Layout #12").hasMatch());
    EXPECT_FALSE(layout.match("Layout #12\n").hasMatch());
    EXPECT_FALSE(layout.match("prefix Layout #12").hasMatch());
    EXPECT_TRUE(qtCompatRegExp("[$]").match("$").hasMatch());
    EXPECT_TRUE(qtCompatRegExp("\\$").match("$").hasMatch());
    EXPECT_EQ((QStringList{QString(), word, QStringLiteral("x"), QString()}),
              qtCompatSplitRegex(" " + word + " x ", "\\s+"));
}

TEST(QtCompat, RegExpKeepsCapturesAndReplacementSemantics) {
    const auto regex = qtCompatRegExp(QStringLiteral("([a-z]+):(\\d+)"));
    const auto match = regex.match(QStringLiteral("tag:42"));
    ASSERT_TRUE(match.hasMatch());
    EXPECT_EQ(0, match.capturedStart());
    EXPECT_EQ(QStringLiteral("42"), match.captured(2));
    EXPECT_EQ(4, match.capturedStart(2));
    EXPECT_EQ(6, match.capturedLength());
    EXPECT_FALSE(regex.match("missing").hasMatch());
    EXPECT_EQ(QStringLiteral("42"), match.captured(2));
    const QString source = QStringLiteral("ab ab");
    EXPECT_EQ(QStringLiteral("ba ba"),
              qtCompatReplace(source, qtCompatRegExp("(a)(b)"), "\\2\\1"));
    QString mutableSource = QStringLiteral("ab");
    EXPECT_EQ(QStringLiteral("-a-b-"),
              qtCompatReplace(mutableSource, qtCompatRegExp(""), "-"));
    EXPECT_EQ(QStringLiteral("-a-b-"), mutableSource);

    // CloudCompare c7d5bb7: both matrix dialogs remove nonempty annotations.
    const auto brackets = qtCompatRegExp(R"(\[([^]]+)\])");
    EXPECT_EQ(QStringLiteral("1  0  0"),
              qtCompatReplace(QStringLiteral("1 [row] 0 [column] 0"), brackets,
                              ""));
    EXPECT_EQ(
            QStringLiteral("1  0"),
            qtCompatReplace(QStringLiteral("1 [row\ncolumn] 0"), brackets, ""));
    EXPECT_EQ(QStringLiteral("[] [unfinished"),
              qtCompatReplace(QStringLiteral("[] [unfinished"), brackets, ""));
}

TEST(QtCompat, NativeCodecsKeepIndependentStateAndDeclaredControls) {
    const QByteArray bytes("A\0B", 3);
    const QString text = QString::fromLatin1(bytes.constData(), bytes.size());
    for (const char* name : {"GB18030", "Shift-JIS", "Windows-1252"}) {
        SCOPED_TRACE(name);
        auto* codec = qtCompatCodecForName(name);
        ASSERT_NE(nullptr, codec);
        EXPECT_EQ(bytes, codec->fromUnicode(text));
        EXPECT_EQ(text, codec->toUnicode(bytes));
        EXPECT_EQ(text, codec->toUnicode(bytes.constData(), bytes.size()));
        EXPECT_EQ(codec, qtCompatCodecForName(name));
        EXPECT_EQ(QByteArray(), codec->fromUnicode(QString(1, QChar(0xd800))));
    }
    auto* western = qtCompatCodecForName("Windows-1252");
    EXPECT_EQ(QString(QChar(0x81)),
              western->toUnicode(QByteArray::fromHex("81")));
    EXPECT_EQ(QByteArray::fromHex("81"),
              western->fromUnicode(QString(QChar(0x81))));
    auto* japanese = qtCompatCodecForName("ISO-2022-JP");
    ASSERT_NE(nullptr, japanese);
    const QString kanji = QString::fromUtf8("\xe4\xb8\xad");
    const QByteArray encoded = japanese->fromUnicode(kanji);
    EXPECT_EQ(kanji, japanese->toUnicode(encoded));
    EXPECT_EQ(encoded, japanese->fromUnicode(kanji));
    EXPECT_EQ(kanji, japanese->toUnicode(encoded));
    const QString waves =
            QString::fromUtf8("\xe3\x80\x9c\xef\xbd\x9e\xc2\xa5\\");
    EXPECT_EQ(QByteArray::fromHex("3f81603f5c"),
              qtCompatCodecForName("Shift-JIS")->fromUnicode(waves));
    EXPECT_EQ(QByteArray::fromHex("3f1b244221411b284a5c1b28425c"),
              japanese->fromUnicode(waves));
    EXPECT_EQ(QString(1, QChar(0xff5e)),
              japanese->toUnicode(QByteArray::fromHex("1b24422141")));
    EXPECT_EQ(QString(1, QChar(0xfffd)),
              qtCompatCodecForName("Shift-JIS")
                      ->toUnicode(QByteArray::fromHex("8241")));

    EXPECT_EQ(nullptr, qtCompatCodecForName(nullptr));
}
