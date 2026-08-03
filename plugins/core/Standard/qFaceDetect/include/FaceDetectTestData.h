// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>

#include <QString>
#include <QVector>

/** One gallery enrollment image parsed from FriendsFaces label.txt. */
struct FaceDetectGalleryEntry {
    QString name;
    QString imagePath;
};

/** FriendsFaces release bundle paths under ~/cloudViewer_data/. */
struct FaceDetectFriendsBundle {
    QString extractRoot;
    QString videoPath;
    QString registerName;
    QString registerImage;
    QString authProbeImage;
    QString batchImage;
    QString registryDbPath;
    QVector<FaceDetectGalleryEntry> galleryEntries;

    bool isUsableForLive() const { return !videoPath.isEmpty(); }
    bool isUsableForRegistry() const {
        return (!registerImage.isEmpty() && !authProbeImage.isEmpty()) ||
               !galleryEntries.isEmpty();
    }
};

namespace FaceDetectTestData {

QString cloudViewerDataRoot();
QString downloadDir();
QString extractDir();
QString zipPath();
QString downloadUrl();
/** Lowercase hex MD5 of the official friends_faces.zip release. */
QString expectedZipMd5();
/** Size + MD5 check; false when missing, truncated, or corrupted. */
bool verifyZipFile(const QString& zipPath);

/** True when extract folder contains a usable bundle (manifest or heuristics). */
bool resolveBundle(FaceDetectFriendsBundle* out);

/** Number of entries inside a zip (0 if unreadable). Used for extract progress. */
int zipEntryCount(const QString& zipPath);

/** Extract zip into extractParentDir (minizip logic from ExtractZIP.cpp). */
using ExtractProgressFn = std::function<void(int current, int total)>;
bool extractZip(const QString& zipPath, const QString& extractParentDir,
                const ExtractProgressFn& onProgress = {});

/** All gallery/label.txt lines (multiple paths per name allowed). */
QVector<FaceDetectGalleryEntry> loadGalleryEntries(const QString& bundleRoot);

/** Query-folder portrait images (Rachel.png, etc.) for auth probes. */
QVector<FaceDetectGalleryEntry> queryPortraitEntries(const QString& bundleRoot);

/** Query-folder frontal portrait for one cast member (query/Rachel.png, …). */
QString queryPortraitPath(const QString& bundleRoot, const QString& name);

/** Curated gallery frontal enrollment photo for FriendsFaces cast (gallery/Name/…). */
QString fixedRegistrationImagePath(const QString& bundleRoot, const QString& name);

/** One enrollment image per Friends cast member (fixed gallery frontals). */
QVector<FaceDetectGalleryEntry> registrationEntriesForBundle(
        const QString& bundleRoot);

/** Default registry DB path under bundle: face_registry_<model>.db */
QString registryPathForModel(const QString& bundleRoot,
                             const QString& modelFilename);

bool isZipCached(qint64 minBytes = 30 * 1024 * 1024);

/** True when path lives under the FriendsFaces sample bundle. */
bool isFriendsBundlePath(const QString& path);

/** QSettings key for the active face registry DB (legacy; prefer manualRegistryDbSettingsKey). */
QString activeRegistrySettingsKey();

/** QSettings keys for user-chosen paths only (never written by Use test data). */
QString manualLiveVideoSettingsKey();
QString manualBatchImageSettingsKey();
QString manualRegistryDbSettingsKey();

/** Remove stale friends_faces paths from QSettings (no-op when already clean). */
void purgeFriendsPathsFromSettings();

/** Open registry DB and return enrolled identity count (0 if missing/invalid). */
int registryEntryCount(const QString& dbPath);

/** Find a non-empty face_registry_*.db under the FriendsFaces bundle. */
QString discoverRegistryDbPath(const QString& modelFilename);

/** Group photo for Detect/Analyze/Dense batch test data (never a cropped portrait). */
QString groupPhotoPath(const FaceDetectFriendsBundle& bundle);

/** Portrait pair for Verify batch test data (Image A / Image B). */
bool verifyTestImagePair(const FaceDetectFriendsBundle& bundle, QString* imageA,
                         QString* imageB);

}  // namespace FaceDetectTestData
