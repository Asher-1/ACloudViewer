// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvTestDataRepository.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <functional>

#include "ecvModelDownloader.h"

extern "C" {
#include "ioapi.h"
#include "unzip.h"
}

#include <memory>

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

namespace {

// Monstree dataset. Content anchor verified once at ingestion (streamed
// digest) and recorded in the <zip>.cvintegrity ledger; later access
// checks are stat-only. SHA-256 pins were computed from MD5-verified
// release artifacts.
constexpr const char* kMonstreeZipName = "dataset_monstree.zip";
constexpr const char* kMonstreeExtractDir = "dataset_monstree";
constexpr const char* kMonstreeDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "reconstruction_data/dataset_monstree.zip";
constexpr const char* kMonstreeSha256 =
        "db890a8f64780ef3c491a088a3be9d34fa4cbb26eeca31394e715b0cd4cce46c";

// FriendsFaces dataset
constexpr const char* kFriendsZipName = "friends_faces.zip";
constexpr const char* kFriendsExtractDir = "friends_faces";
constexpr const char* kFriendsDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "qFaceDetect/friends_faces.zip";
constexpr const char* kFriendsSha256 =
        "eb2c2daff249f8bf50e9f95ae1a37c3ce7de06bc7e2a798c4512a179265a637b";

// Shared object detection / background removal / line detection samples.
// SHA-256 computed from the MD5-pinned release artifact
// (78b6cfa17cdcb99a54dda160b242a52f, 62381393 bytes).
constexpr const char* kObjectsDetectionZipName = "objects_detection_data.zip";
constexpr const char* kObjectsDetectionExtractDir = "objects_detection_data";
constexpr const char* kObjectsDetectionDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "objects_detection_data/objects_detection_data.zip";
constexpr const char* kObjectsDetectionSha256 =
        "dec2c84dff7adefe992291533952f2314ff867a7c35cd705470c622d27515ef5";

// Single-image-to-3D samples (qTrellis): 33 curated images in examples_images/
// plus multi-view (mv/), texture (example_texturing/), HDRI and webp extras.
constexpr const char* kImage2MeshZipName = "image_to_mesh_data.zip";
constexpr const char* kImage2MeshExtractDir = "image_to_mesh_data";
constexpr const char* kImage2MeshDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "Image2MeshData/image_to_mesh_data.zip";
constexpr const char* kImage2MeshSha256 =
        "3a6f4c4156f5b4554f7a002dc898d7a09b2061300e205c06230400e44d65cf41";

// SAM3 segmentation samples (qSAM3): 14 images in images/ + 7 tracking
// videos in videos/.
constexpr const char* kSam3ZipName = "sam_test_data.zip";
constexpr const char* kSam3ExtractDir = "sam_test_data";
constexpr const char* kSam3DownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "sam_test_data/sam_test_data.zip";
constexpr const char* kSam3Sha256 =
        "3c1d97fddc540dfbc134738aa29fa1607bba329c03aa5c0df37f1e6e13317e87";

// Official GKDT demo images (qGKD): the 15 images behind the upstream
// General-Keypoint-Detection README examples, incl. the text-prompt demo
// (2007_007524.jpg) and its 1-shot support image (2007_003778.jpg).
constexpr const char* kGkdZipName = "general_keypoint_detection_data.zip";
constexpr const char* kGkdExtractDir = "general_keypoint_detection_data";
constexpr const char* kGkdDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "general_keypoint_detection_data/general_keypoint_detection_data.zip";
constexpr const char* kGkdSha256 =
        "14da5f5296fcac60a248cde7623f8d93d9fa6b29b8cc9a94a0258f5da4189e23";

// LingBot-Map streaming reconstruction scenes (qLingbotMap). Four official
// demo sequences (courthouse/loop/oxford/university, pre-cropped to the
// 518-wide processing grid) plus the cached native sky masks for the two
// outdoor scenes that ship them. SHA-256 pins come from the
// lingbot_map_data release (expanded_assets digest markers).
constexpr const char* kLingbotMapDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "lingbot_map_data/";
constexpr const char* kLingbotMapCourthouseZipName = "courthouse.zip";
constexpr const char* kLingbotMapCourthouseExtractDir = "courthouse";
constexpr const char* kLingbotMapCourthouseSha256 =
        "fd0a19083600a5d588fdbb4740c808b5ca227ad99d48f2760f65e7bb6aa34ed4";
// Official long-model demo videos (single-file mp4 assets: the downloaded
// file is the content, no extraction). SHA-256 pinned from the release
// assets (locally verified against the upstream long_real campaign inputs).
constexpr const char* kLingbotMapDriveVideoName = "drive_frames.mp4";
constexpr const char* kLingbotMapDriveVideoExtractDir = "lingbot_map_long";
constexpr const char* kLingbotMapDriveVideoSha256 =
        "da814b6ca859d189c5e636ddbadfcc7e14ee532eed904d06b5f3e20de79497d9";
constexpr const char* kLingbotMapLingboWorldVideoName =
        "lingbo_world_frames.mp4";
constexpr const char* kLingbotMapLingboWorldVideoExtractDir =
        "lingbot_map_long";
constexpr const char* kLingbotMapLingboWorldVideoSha256 =
        "28bd8cbb7cf6b214865176c74083e89ed40adba50add05a768adae990913478e";
constexpr const char* kLingbotMapLoopZipName = "loop.zip";
constexpr const char* kLingbotMapLoopExtractDir = "loop";
constexpr const char* kLingbotMapLoopSha256 =
        "67b49e8c4a56108b80700da3da6dc303f694ab4f028337625f57725eb573f3de";
constexpr const char* kLingbotMapOxfordZipName = "oxford.zip";
constexpr const char* kLingbotMapOxfordExtractDir = "oxford";
constexpr const char* kLingbotMapOxfordSha256 =
        "a81a975c80fc7f9e44224730c78dc3d3f3c52f1ca2200c13b64b57a130cbb032";
constexpr const char* kLingbotMapOxfordSkyMasksZipName = "oxford_sky_masks.zip";
constexpr const char* kLingbotMapOxfordSkyMasksExtractDir = "oxford_sky_masks";
constexpr const char* kLingbotMapOxfordSkyMasksSha256 =
        "8f8d7a6981ac8b56e9a2cd9665774a554901d0d0060caeb2fd9f873e21c02c27";
constexpr const char* kLingbotMapUniversityZipName = "university.zip";
constexpr const char* kLingbotMapUniversityExtractDir = "university";
constexpr const char* kLingbotMapUniversitySha256 =
        "bd745b5847ae8d2a5b1385e9e19a6f5de79dda80405864b42f2dee2e8492f653";
constexpr const char* kLingbotMapUniversitySkyMasksZipName =
        "university_sky_masks.zip";
constexpr const char* kLingbotMapUniversitySkyMasksExtractDir =
        "university_sky_masks";
constexpr const char* kLingbotMapUniversitySkyMasksSha256 =
        "fccfd832ea675d7b29708c80ed37504c1f5d1f5e6ab19bf8eb3cbf620498324e";

}  // namespace

// ----------------------------------------------------------------------------
// Static path helpers
// ----------------------------------------------------------------------------

QString ecvTestDataRepository::dataRoot() {
    return QDir(QDir::homePath()).filePath(QStringLiteral("cloudViewer_data"));
}

QString ecvTestDataRepository::downloadDir() {
    return QDir(dataRoot()).filePath(QStringLiteral("download"));
}

QString ecvTestDataRepository::extractDir() {
    return QDir(dataRoot()).filePath(QStringLiteral("extract"));
}

QString ecvTestDataRepository::zipPath(Dataset kind) {
    const auto info = getDatasetInfo(kind);
    return QDir(downloadDir()).filePath(info.zipFileName);
}

QString ecvTestDataRepository::extractPath(Dataset kind) {
    const auto info = getDatasetInfo(kind);
    return QDir(extractDir()).filePath(info.extractDirName);
}

QString ecvTestDataRepository::findDatasetFile(Dataset kind,
                                               const QString& fileName) {
    if (fileName.isEmpty() || QFileInfo(fileName).fileName() != fileName) {
        return {};
    }

    const QString root = extractPath(kind);
    if (!QDir(root).exists()) return {};

    const QString direct = QDir(root).filePath(fileName);
    if (QFileInfo(direct).isFile()) {
        return QFileInfo(direct).absoluteFilePath();
    }

    QString match;
    QDirIterator it(root, QStringList{fileName}, QDir::Files,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString candidate = QFileInfo(it.next()).absoluteFilePath();
        if (!match.isEmpty()) {
            return {};  // Ambiguous archive contents must not pick arbitrarily.
        }
        match = candidate;
    }
    return match;
}

// ----------------------------------------------------------------------------
// Dataset metadata
// ----------------------------------------------------------------------------

ecvTestDataRepository::DatasetInfo ecvTestDataRepository::getDatasetInfo(
        Dataset kind) {
    switch (kind) {
        case Dataset::Monstree:
            return {kind,
                    QStringLiteral("Monstree"),
                    QString::fromLatin1(kMonstreeZipName),
                    QString::fromLatin1(kMonstreeExtractDir),
                    QString::fromLatin1(kMonstreeDownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kMonstreeSha256)}};
        case Dataset::FriendsFaces:
            return {kind,
                    QStringLiteral("FriendsFaces"),
                    QString::fromLatin1(kFriendsZipName),
                    QString::fromLatin1(kFriendsExtractDir),
                    QString::fromLatin1(kFriendsDownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kFriendsSha256)}};
        case Dataset::ObjectsDetection:
            return {kind,
                    QStringLiteral("ObjectsDetection"),
                    QString::fromLatin1(kObjectsDetectionZipName),
                    QString::fromLatin1(kObjectsDetectionExtractDir),
                    QString::fromLatin1(kObjectsDetectionDownloadUrl),
                    {QCryptographicHash::Sha256,
                     QByteArray(kObjectsDetectionSha256)}};
        case Dataset::Image2Mesh:
            return {kind,
                    QStringLiteral("Image2Mesh"),
                    QString::fromLatin1(kImage2MeshZipName),
                    QString::fromLatin1(kImage2MeshExtractDir),
                    QString::fromLatin1(kImage2MeshDownloadUrl),
                    {QCryptographicHash::Sha256,
                     QByteArray(kImage2MeshSha256)}};
        case Dataset::SAM3:
            return {kind,
                    QStringLiteral("SAM3"),
                    QString::fromLatin1(kSam3ZipName),
                    QString::fromLatin1(kSam3ExtractDir),
                    QString::fromLatin1(kSam3DownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kSam3Sha256)}};
        case Dataset::GeneralKeypointDetection:
            return {kind,
                    QStringLiteral("GeneralKeypointDetection"),
                    QString::fromLatin1(kGkdZipName),
                    QString::fromLatin1(kGkdExtractDir),
                    QString::fromLatin1(kGkdDownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kGkdSha256)}};
        case Dataset::LingbotMapCourthouse:
            return {kind,
                    QStringLiteral("LingBot-Map courthouse (outdoor)"),
                    QString::fromLatin1(kLingbotMapCourthouseZipName),
                    QString::fromLatin1(kLingbotMapCourthouseExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapCourthouseZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapCourthouseSha256)}};
        case Dataset::LingbotMapLoop:
            return {kind,
                    QStringLiteral("LingBot-Map loop (indoor loop closure)"),
                    QString::fromLatin1(kLingbotMapLoopZipName),
                    QString::fromLatin1(kLingbotMapLoopExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapLoopZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapLoopSha256)}};
        case Dataset::LingbotMapOxford:
            return {kind,
                    QStringLiteral("LingBot-Map Oxford Spires (outdoor)"),
                    QString::fromLatin1(kLingbotMapOxfordZipName),
                    QString::fromLatin1(kLingbotMapOxfordExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapOxfordZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapOxfordSha256)}};
        case Dataset::LingbotMapOxfordSkyMasks:
            return {kind,
                    QStringLiteral("LingBot-Map oxford sky masks"),
                    QString::fromLatin1(kLingbotMapOxfordSkyMasksZipName),
                    QString::fromLatin1(kLingbotMapOxfordSkyMasksExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(
                                    kLingbotMapOxfordSkyMasksZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapOxfordSkyMasksSha256)}};
        case Dataset::LingbotMapUniversity:
            return {kind,
                    QStringLiteral("LingBot-Map university (outdoor)"),
                    QString::fromLatin1(kLingbotMapUniversityZipName),
                    QString::fromLatin1(kLingbotMapUniversityExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapUniversityZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapUniversitySha256)}};
        case Dataset::LingbotMapUniversitySkyMasks:
            return {kind,
                    QStringLiteral("LingBot-Map university sky masks"),
                    QString::fromLatin1(kLingbotMapUniversitySkyMasksZipName),
                    QString::fromLatin1(
                            kLingbotMapUniversitySkyMasksExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(
                                    kLingbotMapUniversitySkyMasksZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapUniversitySkyMasksSha256)}};
        case Dataset::LingbotMapDriveVideo:
            // Single-file mp4 asset: zipFileName carries the download file,
            // extractDirName the directory it is materialized into.
            return {kind,
                    QStringLiteral("LingBot-Map drive (long model)"),
                    QString::fromLatin1(kLingbotMapDriveVideoName),
                    QString::fromLatin1(kLingbotMapDriveVideoExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapDriveVideoName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapDriveVideoSha256)}};
        case Dataset::LingbotMapLingboWorldVideo:
            return {kind,
                    QStringLiteral("LingBot-Map lingbo world (long model)"),
                    QString::fromLatin1(kLingbotMapLingboWorldVideoName),
                    QString::fromLatin1(kLingbotMapLingboWorldVideoExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(
                                    kLingbotMapLingboWorldVideoName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapLingboWorldVideoSha256)}};
    }
    Q_UNREACHABLE();
    return {};
}

// ----------------------------------------------------------------------------
// Singleton
// ----------------------------------------------------------------------------

ecvTestDataRepository& ecvTestDataRepository::instance() {
    static ecvTestDataRepository s_instance;
    return s_instance;
}

ecvTestDataRepository::ecvTestDataRepository(QObject* parent)
    : QObject(parent), m_downloader(new ecvModelDownloader(this)) {
    connect(m_downloader, &ecvModelDownloader::progress, this,
            &ecvTestDataRepository::onDownloaderProgress);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            &ecvTestDataRepository::onDownloaderFinished);
    // Surface the transport-level detail (resume attempts, stall aborts,
    // the actual Qt error string) that the wrapper messages below omit.
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            [this](const QString& message) {
                emit downloadLogMessage(message);
            });
}

ecvTestDataRepository::~ecvTestDataRepository() = default;

// ----------------------------------------------------------------------------
// Availability query
// ----------------------------------------------------------------------------

bool ecvTestDataRepository::isDatasetAvailable(Dataset kind) const {
    // A directory alone is not a valid cache marker: an interrupted extract
    // leaves the root behind. Validate the files that each consumer needs so
    // a partial cache can fall through to the intact zip or a fresh download.
    const QString extract = extractPath(kind);
    bool extractedComplete = false;
    switch (kind) {
        case Dataset::Monstree:
            extractedComplete = !getMonstreeImages(extract).isEmpty();
            break;
        case Dataset::FriendsFaces:
            extractedComplete = !findFriendsVideo(extract).isEmpty();
            break;
        case Dataset::ObjectsDetection: {
            const QStringList required = {
                    QStringLiteral("bus.jpg"),
                    QStringLiteral("000000397133.jpg"),
                    QStringLiteral("cat.jpg"),
                    QStringLiteral("aerial_airport.jpg"),
                    QStringLiteral("deeplsd_examples.jpg"),
                    QStringLiteral("supervision_demo.mp4"),
                    QStringLiteral("traffic.mp4")};
            extractedComplete = true;
            for (const QString& fileName : required) {
                if (findDatasetFile(kind, fileName).isEmpty()) {
                    extractedComplete = false;
                    break;
                }
            }
            break;
        }
        case Dataset::Image2Mesh:
            // The main single-image-to-3D samples live in examples_images/.
            extractedComplete = !getImage2MeshImages(extract).isEmpty();
            break;
        case Dataset::SAM3:
            // Both the segmentation images and the tracking videos must be
            // present for the qSAM3 sample-data flow to work.
            extractedComplete = !getSamImages(extract).isEmpty() &&
                                !getSamVideos(extract).isEmpty();
            break;
        case Dataset::GeneralKeypointDetection:
            // The official GKDT demo images must all be present.
            extractedComplete =
                    !getGeneralKeypointDetectionImages(extract).isEmpty();
            break;
        case Dataset::LingbotMapCourthouse:
        case Dataset::LingbotMapLoop:
        case Dataset::LingbotMapOxford:
        case Dataset::LingbotMapUniversity:
        case Dataset::LingbotMapOxfordSkyMasks:
        case Dataset::LingbotMapUniversitySkyMasks:
            // A LingBot-Map bundle is complete when its ordered frame
            // sequence (or mask PNGs) is present.
            extractedComplete = !getLingbotMapImages(extract).isEmpty();
            break;
        case Dataset::LingbotMapDriveVideo:
        case Dataset::LingbotMapLingboWorldVideo:
            // Single-file mp4 datasets: complete when the video is
            // materialized under the extract directory.
            extractedComplete =
                    !findDatasetFile(kind, getDatasetInfo(kind).zipFileName)
                             .isEmpty();
            break;
    }
    if (extractedComplete) return true;

    // Check if a verified zip is cached. DeepVerify self-heals archives
    // downloaded before the integrity ledger existed (one hash pass, then
    // the state is recorded and later checks are stat-only).
    const auto info = getDatasetInfo(kind);
    const QString zip = zipPath(kind);
    return ecvAssetIntegrity::isVerified(zip, info.anchor, 0, false,
                                         ecvAssetIntegrity::OnMiss::DeepVerify);
}

// ----------------------------------------------------------------------------
// Download
// ----------------------------------------------------------------------------

void ecvTestDataRepository::startDownload(Dataset kind) {
    if (m_downloadInProgress) {
        emit downloadLogMessage(
                QStringLiteral("[Warning] Download already in progress"));
        return;
    }

    const auto info = getDatasetInfo(kind);
    m_currentDataset = kind;

    // Ensure directories exist
    QDir().mkpath(downloadDir());
    QDir().mkpath(extractDir());

    const QString destPath = zipPath(kind);

    // Check if already downloaded and verified
    if (ecvAssetIntegrity::isVerified(destPath, info.anchor, 0, false,
                                      ecvAssetIntegrity::OnMiss::DeepVerify)) {
        emit downloadLogMessage(
                QStringLiteral("[Info] %1 dataset already downloaded")
                        .arg(info.displayName));
        m_downloadInProgress = false;
        emit downloadFinished(true, kind);
        return;
    }

    // Remove invalid cached file (and its ledger, so the artifact and the
    // recorded evidence never diverge).
    if (QFileInfo::exists(destPath)) {
        QFile::remove(destPath);
        ecvAssetIntegrity::invalidate(destPath);
    }

    m_downloadInProgress = true;
    emit downloadLogMessage(QStringLiteral("[Info] Downloading %1 dataset...")
                                    .arg(info.displayName));

    ecvModelDownloader::Request request;
    request.url = info.downloadUrl;
    request.destPath = destPath;
    request.minBytes = 1024 * 1024;       // At least 1 MB
    request.requireGgufMagic = false;     // Not a GGUF file
    request.contentAnchor = info.anchor;  // streamed digest check at ingestion

    m_downloader->download(request);
}

void ecvTestDataRepository::cancelDownload() {
    if (m_downloadInProgress && m_downloader) {
        m_downloader->cancel();
        m_downloadInProgress = false;
    }
}

void ecvTestDataRepository::onDownloaderProgress(qint64 received,
                                                 qint64 total) {
    if (total <= 0) return;
    const int percent = static_cast<int>((received * 100) / total);
    const QString status =
            QStringLiteral("%1 / %2 (%3%)")
                    .arg(ecvModelDownloader::formatFileSize(received),
                         ecvModelDownloader::formatFileSize(total))
                    .arg(percent);
    emit downloadProgress(percent, status);
}

void ecvTestDataRepository::onDownloaderFinished(bool ok,
                                                 const QString& destPath) {
    m_downloadInProgress = false;
    const auto info = getDatasetInfo(m_currentDataset);

    if (!ok) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Failed to download %1 dataset")
                        .arg(info.displayName));
        emit downloadFinished(false, m_currentDataset);
        return;
    }

    // Content verification already happened inside the downloader (the
    // digest was streamed while writing) and the verified state is
    // recorded in the zip's integrity ledger — nothing to re-hash here.
    emit downloadLogMessage(
            QStringLiteral("[Info] Downloaded %1 dataset successfully")
                    .arg(info.displayName));
    emit downloadFinished(true, m_currentDataset);
}

// ----------------------------------------------------------------------------
// Extraction (using minizip for cross-platform consistency)
// ----------------------------------------------------------------------------

namespace {

constexpr int kExtractBufferSize = 8192;
constexpr int kMaxZipEntryNameLen = 1024;

// RAII wrapper for minizip unzFile
struct UnzFileCloser {
    void operator()(unzFile uf) const {
        if (uf) unzClose(uf);
    }
};
using UniqueUnzFile = std::unique_ptr<void, UnzFileCloser>;

#ifdef Q_OS_WIN
// Windows: _wfopen supports Unicode paths (UTF-16), unlike fopen which uses
// the ANSI code page. minizip's unzOpen64 internally calls fopen, so we
// provide a custom file function table that uses _wfopen.
static voidpf ZCALLBACK wfopen64_file_func(voidpf /*opaque*/,
                                           const void* filename,
                                           int mode) {
    const wchar_t* wpath = static_cast<const wchar_t*>(filename);
    const wchar_t* wmode = nullptr;
    if ((mode & ZLIB_FILEFUNC_MODE_READWRITEFILTER) == ZLIB_FILEFUNC_MODE_READ)
        wmode = L"rb";
    else if (mode & ZLIB_FILEFUNC_MODE_EXISTING)
        wmode = L"r+b";
    else if (mode & ZLIB_FILEFUNC_MODE_CREATE)
        wmode = L"wb";
    if (wpath && wmode) return _wfopen(wpath, wmode);
    return nullptr;
}
#endif

UniqueUnzFile openUnzFile(const QString& path) {
#ifdef Q_OS_WIN
    // Pass wide-string path so our custom open func can use _wfopen
    zlib_filefunc64_def filefunc;
    fill_fopen64_filefunc(&filefunc);
    filefunc.zopen64_file = wfopen64_file_func;
    const std::wstring wpath = path.toStdWString();
    return UniqueUnzFile(unzOpen2_64(wpath.c_str(), &filefunc));
#else
    // macOS/Linux: fopen uses UTF-8 natively
    return UniqueUnzFile(unzOpen64(path.toUtf8().constData()));
#endif
}

int extractCurrentZipEntry(unzFile uf, const QString& extractDir) {
    char filename_inzip[kMaxZipEntryNameLen] = {};

    unz_file_info64 file_info;
    int err = unzGetCurrentFileInfo64(uf, &file_info, filename_inzip,
                                      sizeof(filename_inzip), nullptr, 0,
                                      nullptr, 0);
    if (err != UNZ_OK) return err;

    const QString entryName = QString::fromUtf8(filename_inzip);

    // Security: reject path traversal attempts
    if (entryName.contains(QStringLiteral("..")) ||
        entryName.startsWith(QLatin1Char('/')) ||
        entryName.startsWith(QLatin1Char('\\'))) {
        return UNZ_ERRNO;
    }

    // Directory entry — just create it
    if (entryName.endsWith(QLatin1Char('/'))) {
        QDir().mkpath(QDir(extractDir).filePath(entryName));
        return UNZ_OK;
    }

    // File entry — open, read, write
    err = unzOpenCurrentFilePassword(uf, nullptr);
    if (err != UNZ_OK) return err;

    const QString filePath = QDir(extractDir).filePath(entryName);
    QDir().mkpath(QFileInfo(filePath).path());

    QFile outFile(filePath);
    if (!outFile.open(QIODevice::WriteOnly)) {
        unzCloseCurrentFile(uf);
        return UNZ_ERRNO;
    }

    QByteArray buf(kExtractBufferSize, Qt::Uninitialized);
    do {
        err = unzReadCurrentFile(uf, buf.data(), buf.size());
        if (err < 0) break;
        if (err > 0) {
            if (outFile.write(buf.constData(), err) != err) {
                err = UNZ_ERRNO;
                break;
            }
        }
    } while (err > 0);

    outFile.close();

    if (err == UNZ_OK) {
        err = unzCloseCurrentFile(uf);
    } else {
        unzCloseCurrentFile(uf);
    }
    return err;
}

bool extractFromZipFile(const QString& zipPath,
                        const QString& extractDir,
                        const std::function<void(int, int)>& onProgress) {
    if (zipPath.isEmpty()) return false;

    auto uf = openUnzFile(zipPath);
    if (!uf) return false;

    unz_global_info64 gi;
    int err = unzGetGlobalInfo64(uf.get(), &gi);
    if (err != UNZ_OK) return false;

    const int totalEntries = static_cast<int>(gi.number_entry);
    for (uLong i = 0; i < gi.number_entry; ++i) {
        err = extractCurrentZipEntry(uf.get(), extractDir);
        if (err != UNZ_OK) return false;

        if (onProgress && totalEntries > 0) {
            onProgress(static_cast<int>(i + 1), totalEntries);
        }

        if ((i + 1) < gi.number_entry) {
            err = unzGoToNextFile(uf.get());
            if (err != UNZ_OK) return false;
        }
    }

    return true;
}

}  // namespace

// ----------------------------------------------------------------------------
// Static utility methods
// ----------------------------------------------------------------------------

int ecvTestDataRepository::zipEntryCount(const QString& zipPath) {
    if (zipPath.isEmpty() || !QFileInfo::exists(zipPath)) return 0;
    auto uf = openUnzFile(QFileInfo(zipPath).absoluteFilePath());
    if (!uf) return 0;
    unz_global_info64 gi;
    const int err = unzGetGlobalInfo64(uf.get(), &gi);
    return err == UNZ_OK ? static_cast<int>(gi.number_entry) : 0;
}

bool ecvTestDataRepository::extractZip(const QString& zipPath,
                                       const QString& extractDir,
                                       const ExtractProgressFn& onProgress) {
    if (zipPath.isEmpty() || !QFileInfo::exists(zipPath)) return false;
    QDir().mkpath(extractDir);
    return extractFromZipFile(QFileInfo(zipPath).absoluteFilePath(), extractDir,
                              onProgress);
}

bool ecvTestDataRepository::isSingleFileDataset(Dataset kind) {
    switch (kind) {
        case Dataset::LingbotMapDriveVideo:
        case Dataset::LingbotMapLingboWorldVideo:
            return true;
        default:
            return false;
    }
}

bool ecvTestDataRepository::extractDataset(Dataset kind) {
    const auto info = getDatasetInfo(kind);
    const QString zip = zipPath(kind);
    const QString extract = extractDir();

    if (!ecvAssetIntegrity::isVerified(zip, info.anchor, 0, false,
                                       ecvAssetIntegrity::OnMiss::DeepVerify)) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Zip file is missing or invalid: %1")
                        .arg(zip));
        emit extractionFinished(false, kind);
        return false;
    }

    // Single-file datasets (e.g. the LingBot-Map long-model mp4s): the
    // downloaded file itself is the content. Materialize it under the
    // extract directory and skip the zip machinery entirely.
    if (isSingleFileDataset(kind)) {
        if (!QDir().mkpath(extract)) {
            emit downloadLogMessage(
                    QStringLiteral(
                            "[Error] Cannot create extract directory: %1")
                            .arg(extract));
            emit extractionFinished(false, kind);
            return false;
        }
        const QString dest = QDir(extract).filePath(info.zipFileName);
        QFile::remove(dest);
        if (!QFile::copy(zip, dest)) {
            emit downloadLogMessage(
                    QStringLiteral("[Error] Cannot materialize %1 -> %2")
                            .arg(zip, dest));
            emit extractionFinished(false, kind);
            return false;
        }
        emit extractionProgress(1, 1);
        emit extractionFinished(true, kind);
        return true;
    }

    // Ensure extract directory exists
    if (!QDir().mkpath(extract)) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Cannot create extract directory: %1")
                        .arg(extract));
        emit extractionFinished(false, kind);
        return false;
    }

    // Use minizip for cross-platform consistency
    auto onProgress = [this](int current, int total) {
        emit extractionProgress(current, total);
    };

    const bool ok = extractZip(zip, extract, onProgress);

    if (!ok) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Extraction failed: %1").arg(zip));
        emit extractionFinished(false, kind);
        return false;
    }

    // Verify extraction result
    const QString expectedDir = extractPath(kind);
    if (!QDir(expectedDir).exists()) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Extraction completed but expected "
                               "directory not found: %1")
                        .arg(expectedDir));
        emit extractionFinished(false, kind);
        return false;
    }

    emit downloadLogMessage(
            QStringLiteral("[Info] Extracted %1 dataset successfully")
                    .arg(info.displayName));
    emit extractionFinished(true, kind);
    return true;
}

// ----------------------------------------------------------------------------
// Dataset-specific helpers
// ----------------------------------------------------------------------------

QStringList ecvTestDataRepository::getMonstreeImages(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // Look specifically in mini3/ subdirectory
    const QString imageDir = QDir(bundleRoot).filePath(QStringLiteral("mini3"));
    if (!QDir(imageDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.jpg"),  QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"),  QStringLiteral("*.tif"),
            QStringLiteral("*.tiff"), QStringLiteral("*.webp")};

    QStringList images;
    QDirIterator it(imageDir, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QString ecvTestDataRepository::findFriendsVideo(const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // Prefer the known path first
    const QString knownRel =
            QDir(bundleRoot).filePath(QStringLiteral("query/friends_demo.mp4"));
    if (QFileInfo::exists(knownRel))
        return QFileInfo(knownRel).absoluteFilePath();

    // Fall back to recursive search in query/ subdirectory
    const QString videoDir = QDir(bundleRoot).filePath(QStringLiteral("query"));
    if (!QDir(videoDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.mp4"), QStringLiteral("*.mov"),
            QStringLiteral("*.avi"), QStringLiteral("*.mkv"),
            QStringLiteral("*.webm")};

    QDirIterator it(videoDir, patterns, QDir::Files,
                    QDirIterator::Subdirectories);
    QString best;
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        // Prefer files with "friend" in the name
        if (fileName.contains(QStringLiteral("friend"), Qt::CaseInsensitive))
            return QFileInfo(path).absoluteFilePath();
        if (best.isEmpty()) best = QFileInfo(path).absoluteFilePath();
    }
    return best;
}

QStringList ecvTestDataRepository::getImage2MeshImages(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // The curated single-image-to-3D samples live in examples_images/.
    const QString imageDir =
            QDir(bundleRoot).filePath(QStringLiteral("examples_images"));
    if (!QDir(imageDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"), QStringLiteral("*.webp")};

    QStringList images;
    QDirIterator it(imageDir, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QStringList ecvTestDataRepository::getGeneralKeypointDetectionImages(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // The official GKDT demo images sit directly in the bundle root.
    if (!QDir(bundleRoot).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"), QStringLiteral("*.webp")};

    QStringList images;
    QDirIterator it(bundleRoot, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QStringList ecvTestDataRepository::getLingbotMapImages(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};
    if (!QDir(bundleRoot).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.png"), QStringLiteral("*.jpg"),
            QStringLiteral("*.jpeg"), QStringLiteral("*.bmp")};

    // The lingbot_map_data archives keep a top-level scene directory
    // (<zip stem>/<frame>.png), so the scan is recursive.
    QStringList images;
    QDirIterator it(bundleRoot, patterns, QDir::Files,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QStringList ecvTestDataRepository::getSamImages(const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    const QString imageDir =
            QDir(bundleRoot).filePath(QStringLiteral("images"));
    if (!QDir(imageDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"), QStringLiteral("*.webp")};

    QStringList images;
    QDirIterator it(imageDir, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QStringList ecvTestDataRepository::getSamVideos(const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    const QString videoDir =
            QDir(bundleRoot).filePath(QStringLiteral("videos"));
    if (!QDir(videoDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.mp4"), QStringLiteral("*.mov"),
            QStringLiteral("*.avi"), QStringLiteral("*.mkv"),
            QStringLiteral("*.webm")};

    QStringList videos;
    QDirIterator it(videoDir, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        videos.append(QFileInfo(path).absoluteFilePath());
    }
    videos.sort(Qt::CaseInsensitive);
    return videos;
}
