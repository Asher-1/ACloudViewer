// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QtCompat.h>
#include <ecvObject.h>
#include <gtest/gtest.h>

#include <QColor>
#include <QDateTime>
#include <QTemporaryFile>
#include <iostream>
#include <limits>

struct RegisteredMetaData {
    QString text;
};
inline QDataStream& operator<<(QDataStream& stream,
                               const RegisteredMetaData& value) {
    return stream << value.text;
}
inline QDataStream& operator>>(QDataStream& stream, RegisteredMetaData& value) {
    return stream >> value.text;
}
Q_DECLARE_METATYPE(RegisteredMetaData)

namespace {
class BinaryObject : public ccObject {
public:
    using ccObject::ccObject;
    using ccObject::fromFile;
    using ccObject::minimumFileVersion;
    using ccObject::toFile;
    CV_CLASS_ENUM getClassID() const override { return CV_TYPES::OBJECT; }
};

bool load(QFile& file, BinaryObject& object, short version = 59) {
    if (!file.seek(0) ||
        ccObject::ReadClassIDFromFile(file, version) != CV_TYPES::OBJECT) {
        return false;
    }
    ccSerializableObject::LoadedIDMap ids;
    return object.fromFile(file, version, 0, ids);
}

// Schema-generated fixture for the old plugin-unavailable fallback; not a
// captured historical user file. Exercise nonempty VLR and raw extra records.
bool writeLegacyLas(QFile& file,
                    quint64 vlrCount = 1,
                    quint64 extraCount = 1,
                    const char* typeName = "LasVlr",
                    quint32 typeTag = 1024) {
    BinaryObject original(QStringLiteral("legacy"));
    if (!original.toFile(file, 58) || !file.seek(file.size() - 4)) return false;
    const quint32 count = 3;
    if (file.write(reinterpret_cast<const char*>(&count), 4) != 4) return false;
    QDataStream stream(&file);
    stream.setVersion(QDataStream::Qt_5_12);
    stream << QStringLiteral("before") << QVariant(QStringLiteral("kept"));
    stream << QStringLiteral("LAS.vlrs") << typeTag << quint8(0);
    stream << typeName;
    stream << vlrCount;
    if (vlrCount == 1) {
        stream << quint16(0);
        stream.writeRawData("0123456789abcdef", 16);
        stream << quint16(7) << quint16(3);
        stream.writeRawData(QByteArray(32, '\0').constData(), 32);
        stream.writeRawData("abc", 3);
    }
    stream << extraCount;
    if (extraCount == 1) {
        stream.writeRawData(QByteArray(272, '\0').constData(), 272);
    }
    stream << QStringLiteral("after") << QVariant(QStringLiteral("aligned"));
    return stream.status() == QDataStream::Ok;
}
}  // namespace

TEST(ObjectSerialization, MetadataUsesQt512WireContract) {
    EXPECT_EQ(QDataStream::Qt_5_12, qtCompatDataStreamVersion());
    BinaryObject original(QString::fromUtf8("A\xc3\xa9\xe4\xb8\xad"));
    original.setUniqueID(0x12345678);
    const QVariantMap values = {
            {QStringLiteral("bool"), true},
            {QStringLiteral("bytes"), QByteArray("A\0B", 3)},
            {QStringLiteral("color"),
             QVariant::fromValue(QColor(12, 34, 56, 78))},
            {QStringLiteral("date"),
             QDateTime(QDate(2020, 2, 29), QTime(12, 34, 56), Qt::UTC)},
            {QStringLiteral("integer"),
             QVariant::fromValue(qint64(-123456789))},
            {QStringLiteral("nested"),
             QVariantMap{{QStringLiteral("text"), original.getName()}}},
    };
    original.setMetaData(values);
    QTemporaryFile file;
    ASSERT_TRUE(file.open());
    ASSERT_TRUE(original.toFile(file, 59));
    ASSERT_TRUE(file.seek(0));
    std::cout << "OBJECT_WIRE|" << file.readAll().toHex().constData() << '\n';

    // Inspect the actual serializer's output, not an unrelated stream fixture.
    ASSERT_TRUE(file.seek(12));  // class ID + unique ID
    QDataStream stream(&file);
    stream.setVersion(QDataStream::Qt_5_12);
    QString name;
    stream >> name;
    EXPECT_EQ(original.getName(), name);
    ASSERT_TRUE(file.seek(file.pos() + 4));  // raw object flags
    quint32 count = 0;
    ASSERT_EQ(4, file.read(reinterpret_cast<char*>(&count), 4));
    EXPECT_EQ(values.size(), count);
    QByteArray expected;
    QDataStream expectedStream(&expected, QIODevice::WriteOnly);
    expectedStream.setVersion(QDataStream::Qt_5_12);
    for (auto it = values.cbegin(); it != values.cend(); ++it) {
        expectedStream << it.key() << it.value();
    }
    EXPECT_EQ(expected, file.readAll());

    BinaryObject restored;
    ASSERT_TRUE(load(file, restored));
    EXPECT_EQ(original.getName(), restored.getName());
    for (auto it = values.cbegin(); it != values.cend(); ++it) {
        EXPECT_EQ(it.value(), restored.getMetaData(it.key()));
    }
}

TEST(ObjectSerialization, RegisteredLasKeyIsRestoredInsteadOfDiscarded) {
    qRegisterMetaType<RegisteredMetaData>();
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    qRegisterMetaTypeStreamOperators<RegisteredMetaData>("RegisteredMetaData");
#endif
    BinaryObject original;
    original.setMetaData("LAS.vlrs",
                         QVariant::fromValue(RegisteredMetaData{"payload"}));
    original.setMetaData("after", QStringLiteral("kept"));
    QTemporaryFile file;
    ASSERT_TRUE(file.open());
    ASSERT_TRUE(original.toFile(file, 59));
    BinaryObject restored;
    ASSERT_TRUE(load(file, restored));
    const QVariant value = restored.getMetaData("LAS.vlrs");
    ASSERT_TRUE(value.canConvert<RegisteredMetaData>());
    EXPECT_EQ(QStringLiteral("payload"),
              value.value<RegisteredMetaData>().text);
    EXPECT_EQ(QStringLiteral("kept"), restored.getMetaData("after").toString());
}

TEST(ObjectSerialization, UnavailableLegacyLasKeepsOtherMetadataAligned) {
    QTemporaryFile file;
    ASSERT_TRUE(file.open());
    ASSERT_TRUE(writeLegacyLas(file));
    BinaryObject restored;
    ASSERT_TRUE(load(file, restored, 58));
    EXPECT_EQ(QStringLiteral("kept"),
              restored.getMetaData("before").toString());
    EXPECT_EQ(QStringLiteral("aligned"),
              restored.getMetaData("after").toString());
    EXPECT_FALSE(restored.hasMetaData("LAS.vlrs"));
}

TEST(ObjectSerialization, LegacyLasRejectsUnboundedCountsAndWrongTypes) {
    const quint64 huge = (std::numeric_limits<quint64>::max)();
    struct Fixture {
        quint64 vlrs;
        quint64 extras;
        const char* type;
    };
    for (const auto& fixture :
         {Fixture{huge, 0, "LasVlr"}, Fixture{0, huge, "LasVlr"},
          Fixture{0, 0, "OtherType"}}) {
        QTemporaryFile file;
        ASSERT_TRUE(file.open());
        ASSERT_TRUE(writeLegacyLas(file, fixture.vlrs, fixture.extras,
                                   fixture.type));
        BinaryObject restored;
        EXPECT_FALSE(load(file, restored));
    }
}

TEST(ObjectSerialization, TruncatedMetadataDoesNotReportSuccess) {
    BinaryObject original;
    original.setMetaData("text", QStringLiteral("not truncated"));
    QTemporaryFile file;
    ASSERT_TRUE(file.open());
    ASSERT_TRUE(original.toFile(file, 59));
    ASSERT_TRUE(file.seek(0));
    const QByteArray bytes = file.readAll();
    for (int size = 0; size < bytes.size(); ++size) {
        QTemporaryFile truncated;
        ASSERT_TRUE(truncated.open());
        ASSERT_EQ(size, truncated.write(bytes.constData(), size));
        BinaryObject restored;
        EXPECT_FALSE(load(truncated, restored)) << "prefix length " << size;
    }
}

TEST(ObjectSerialization, Legacy58RetainsTheCompiledSdkStream) {
    BinaryObject original(QString::fromUtf8("legacy-A\xc3\xa9\xe4\xb8\xad"));
    original.setUniqueID(0x12345678);
    original.setMetaData("color", QVariant::fromValue(QColor(12, 34, 56, 78)));
    original.setMetaData(
            "date", QDateTime(QDate(2020, 2, 29), QTime(12, 34, 56), Qt::UTC));
    original.setMetaData("text", QStringLiteral("unchanged"));
    QTemporaryFile file;
    ASSERT_TRUE(file.open());
    ASSERT_TRUE(original.toFile(file, 58));
    ASSERT_TRUE(file.seek(0));
    std::cout << "LEGACY_OBJECT_WIRE|" << file.readAll().toHex().constData()
              << '\n';
    BinaryObject restored;
    ASSERT_TRUE(load(file, restored, 58));
    EXPECT_EQ(original.getMetaData("color"), restored.getMetaData("color"));
    EXPECT_EQ(original.getMetaData("date"), restored.getMetaData("date"));
    EXPECT_EQ(original.getMetaData("text"), restored.getMetaData("text"));
}

TEST(ObjectSerialization, OnlyPersistentMetadataRaisesMinimumVersion) {
    BinaryObject object;
    EXPECT_EQ(34, object.minimumFileVersion());
    object.setMetaData("transient.nosave", true);
    EXPECT_EQ(34, object.minimumFileVersion());
    object.setMetaData("persisted", true);
    EXPECT_EQ(59, object.minimumFileVersion());
    EXPECT_GE(ccObject::GetCurrentDBVersion(), object.minimumFileVersion());
}

TEST(ObjectSerialization, HistoricalLasTypeTagsAreValidated) {
    for (quint32 tag : {quint32(127), quint32(1024), quint32(65536)}) {
        QTemporaryFile file;
        ASSERT_TRUE(file.open());
        ASSERT_TRUE(writeLegacyLas(file, 1, 1, "LasVlr", tag));
        BinaryObject restored;
        ASSERT_TRUE(load(file, restored, 58)) << tag;
        EXPECT_EQ(QStringLiteral("aligned"),
                  restored.getMetaData("after").toString());
    }
}

TEST(ObjectSerialization, OrdinaryLasKeyValuesArePreserved) {
    for (const auto& value :
         {QVariant(), QVariant(QStringLiteral("ordinary")), QVariant(true)}) {
        BinaryObject source;
        source.setMetaData("LAS.vlrs", value);
        QTemporaryFile file;
        ASSERT_TRUE(file.open());
        ASSERT_TRUE(source.toFile(file, 59));
        BinaryObject restored;
        ASSERT_TRUE(load(file, restored));
        EXPECT_TRUE(restored.hasMetaData("LAS.vlrs"));
        EXPECT_EQ(value, restored.getMetaData("LAS.vlrs"));
    }
}
