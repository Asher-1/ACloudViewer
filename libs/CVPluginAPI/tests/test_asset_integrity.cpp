// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

#include "ecvAssetIntegrity.h"

class AssetIntegrityTest : public ::testing::Test {
protected:
    QTemporaryDir temporary;
    QString path;
    const QByteArray bytes = QByteArray("GGUF") + QByteArray(4096, 'x');
    ecvAssetIntegrity::Anchor pin;
    void SetUp() override {
        ASSERT_TRUE(temporary.isValid());
        path = temporary.filePath("model.gguf");
        pin = {QCryptographicHash::Sha256,
               QCryptographicHash::hash(bytes, QCryptographicHash::Sha256)
                       .toHex()};
        ASSERT_TRUE(write(bytes));
    }
    bool write(const QByteArray& payload) {
        QFile file(path);
        return file.open(QIODevice::WriteOnly | QIODevice::Truncate) &&
               file.write(payload) == payload.size();
    }
    void verifyOriginal() {
        ASSERT_TRUE(ecvAssetIntegrity::isVerified(
                path, pin, 64, true, ecvAssetIntegrity::OnMiss::DeepVerify));
    }
};

TEST_F(AssetIntegrityTest, VerifiedEvidenceSupportsPinnedAndUnpinnedCacheHits) {
    ASSERT_NO_FATAL_FAILURE(verifyOriginal());
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, pin, 64, true));
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, {}, 64, true));
}

TEST_F(AssetIntegrityTest, CorruptionStaysRejectedUntilExplicitRecovery) {
    ASSERT_NO_FATAL_FAILURE(verifyOriginal());
    ASSERT_TRUE(write(bytes + "corrupt"));
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(path, pin, 64, true));
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(path, pin, 64, true));
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(path, {}, 64, true));
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(
            path, pin, 64, true, ecvAssetIntegrity::OnMiss::DeepVerify));
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(path, pin, 64, true));
    EXPECT_TRUE(QFileInfo::exists(path + ".cvintegrity"));
    ASSERT_TRUE(write(bytes));
    ASSERT_NO_FATAL_FAILURE(verifyOriginal());
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, pin, 64, true));
}

TEST_F(AssetIntegrityTest, SourcePinChangesRejectOldPayloadAndEvidence) {
    ASSERT_NO_FATAL_FAILURE(verifyOriginal());
    const ecvAssetIntegrity::Anchor otherPin{QCryptographicHash::Sha256,
                                             QByteArray("11").repeated(32)};
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(path, otherPin, 64, true));
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(path, otherPin, 64, true));
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, pin, 64, true));
}

TEST_F(AssetIntegrityTest, LegacyIngestionStillChecksExactSize) {
    ASSERT_NO_FATAL_FAILURE(verifyOriginal());
    ecvAssetIntegrity::invalidate(path);
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, pin, 64, true));
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(
            path, {}, 64, true, ecvAssetIntegrity::OnMiss::CheapChecksOnly,
            bytes.size() + 1));
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(
            path, {}, 64, true, ecvAssetIntegrity::OnMiss::CheapChecksOnly,
            bytes.size()));
}
