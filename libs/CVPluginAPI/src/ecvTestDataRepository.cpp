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
