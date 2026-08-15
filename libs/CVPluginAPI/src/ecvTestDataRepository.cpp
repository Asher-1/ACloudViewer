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

// Monstree dataset
constexpr const char* kMonstreeZipName = "dataset_monstree.zip";
constexpr const char* kMonstreeExtractDir = "dataset_monstree";
constexpr const char* kMonstreeDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "reconstruction_data/dataset_monstree.zip";
constexpr const char* kMonstreeExpectedMd5 = "10730009514e2db7b47d16f75627561c";
constexpr qint64 kMonstreeExpectedSize = 200 * 1024 * 1024;  // ~206.2 MB

// FriendsFaces dataset
constexpr const char* kFriendsZipName = "friends_faces.zip";
constexpr const char* kFriendsExtractDir = "friends_faces";
constexpr const char* kFriendsDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "qFaceDetect/friends_faces.zip";
constexpr const char* kFriendsExpectedMd5 = "1d1ffebb97edac790b55c6f0f3c9d9fc";
constexpr qint64 kFriendsExpectedSize = 30 * 1024 * 1024;  // ~35 MB

// qManualCalib sample dataset
constexpr const char* kMcalibZipName = "qcalib_test_data.zip";
constexpr const char* kMcalibExtractDir = "qcalib_test_data";
constexpr const char* kMcalibDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "qManualCalib/qcalib_test_data.zip";
constexpr const char* kMcalibExpectedMd5 = "04a458cdd48fc88ae5c878062ca94a80";
constexpr qint64 kMcalibExpectedSize =
        20 * 1024 * 1024;  // ~20 MB (zip ~21.8 MB)

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
                    QString::fromLatin1(kMonstreeExpectedMd5),
                    kMonstreeExpectedSize};
        case Dataset::FriendsFaces:
            return {kind,
                    QStringLiteral("FriendsFaces"),
                    QString::fromLatin1(kFriendsZipName),
                    QString::fromLatin1(kFriendsExtractDir),
                    QString::fromLatin1(kFriendsDownloadUrl),
                    QString::fromLatin1(kFriendsExpectedMd5),
                    kFriendsExpectedSize};
        case Dataset::ManualCalib:
            return {kind,
                    QStringLiteral("ManualCalib"),
                    QString::fromLatin1(kMcalibZipName),
                    QString::fromLatin1(kMcalibExtractDir),
                    QString::fromLatin1(kMcalibDownloadUrl),
                    QString::fromLatin1(kMcalibExpectedMd5),
                    kMcalibExpectedSize};
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
// Integrity verification
// ----------------------------------------------------------------------------

bool ecvTestDataRepository::verifyZipIntegrity(const QString& zipPath,
                                               const QString& expectedMd5,
                                               qint64 expectedMinSize) {
    if (zipPath.isEmpty() || !QFileInfo::exists(zipPath)) return false;

    const QFileInfo fi(zipPath);

    // Check file size first (fast rejection of truncated downloads)
    if (expectedMinSize > 0 && fi.size() < expectedMinSize) return false;

    // If no expected MD5 provided, just check file exists and is non-empty
    if (expectedMd5.isEmpty()) {
        return fi.size() > 0;
    }

    QFile file(zipPath);
    if (!file.open(QIODevice::ReadOnly)) return false;

    QCryptographicHash hash(QCryptographicHash::Md5);
    if (!hash.addData(&file)) return false;
    file.close();

    const QString actual = QString::fromLatin1(hash.result().toHex());
    return actual.compare(expectedMd5, Qt::CaseInsensitive) == 0;
}

bool ecvTestDataRepository::isDatasetAvailable(Dataset kind) const {
    // Check extract dir FIRST — dataset may be extracted even if zip was
    // deleted
    const QString extract = extractPath(kind);
    if (QDir(extract).exists()) return true;

    // Check if a valid zip is cached
    const auto info = getDatasetInfo(kind);
    const QString zip = zipPath(kind);
    return verifyZipIntegrity(zip, info.expectedMd5, info.expectedSize);
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

    // Check if already downloaded and valid (size + MD5)
    if (verifyZipIntegrity(destPath, info.expectedMd5, info.expectedSize)) {
        emit downloadLogMessage(
                QStringLiteral("[Info] %1 dataset already downloaded")
                        .arg(info.displayName));
        m_downloadInProgress = false;
        emit downloadFinished(true, kind);
        return;
    }

    // Remove invalid cached file
    if (QFileInfo::exists(destPath)) {
        QFile::remove(destPath);
    }

    m_downloadInProgress = true;
    emit downloadLogMessage(QStringLiteral("[Info] Downloading %1 dataset...")
                                    .arg(info.displayName));

    ecvModelDownloader::Request request;
    request.url = info.downloadUrl;
    request.destPath = destPath;
    request.minBytes = 1024 * 1024;    // At least 1 MB
    request.requireGgufMagic = false;  // Not a GGUF file

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

    // Verify integrity (size + MD5)
    if (!verifyZipIntegrity(destPath, info.expectedMd5, info.expectedSize)) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Downloaded file failed integrity "
                               "check"));
        QFile::remove(destPath);
        emit downloadFinished(false, m_currentDataset);
        return;
    }

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

    if (!QFileInfo::exists(zip)) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Zip file not found: %1").arg(zip));
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

QString ecvTestDataRepository::getManualCalibBagPath(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // The sample ROS bag is expected under <root>/bags/sample_aligned.bag
    const QString bagDir = QDir(bundleRoot).filePath(QStringLiteral("bags"));
    if (!QDir(bagDir).exists()) return {};

    const QStringList patterns = {QStringLiteral("*.bag")};
    QDirIterator it(bagDir, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        // Prefer the aligned sample bag when present
        if (fileName.contains(QStringLiteral("sample"), Qt::CaseInsensitive))
            return QFileInfo(path).absoluteFilePath();
        return QFileInfo(path).absoluteFilePath();
    }
    return {};
}

QString ecvTestDataRepository::getManualCalibConfigDir(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    const QString configDir =
            QDir(bundleRoot).filePath(QStringLiteral("configs"));
    if (QDir(configDir).exists()) return QDir(configDir).absolutePath();
    return {};
}
