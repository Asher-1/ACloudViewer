// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <functional>

#include "CVPluginAPI.h"

class ecvModelDownloader;

/**
 * @brief Unified test data repository for reconstruction plugins.
 *
 * Provides a single source of truth for downloading, caching, and accessing
 * test datasets (Monstree, FriendsFaces, etc.) used by qFreeSplatter, qDA3,
 * qlightglue, and other reconstruction plugins.
 *
 * Directory structure:
 *   ~/cloudViewer_data/
 *     ├── download/          # Raw zip files
 *     │   ├── dataset_monstree.zip
 *     │   └── friends_faces.zip
 *     └── extract/           # Extracted datasets
 *         ├── dataset_monstree/
 *         └── friends_faces/
 */
class CVPLUGIN_LIB_API ecvTestDataRepository : public QObject {
    Q_OBJECT

public:
    /** Available test datasets. */
    enum class Dataset {
        Monstree,     ///< Monstree dataset for image-based reconstruction
        FriendsFaces  ///< FriendsFaces video for face capture
    };

    /** Dataset metadata. */
    struct DatasetInfo {
        Dataset kind;
        QString displayName;     ///< Human-readable name
        QString zipFileName;     ///< Name of the zip file
        QString extractDirName;  ///< Directory name after extraction
        QString downloadUrl;     ///< Remote URL
        QString expectedMd5;     ///< Expected MD5 hash
        qint64 expectedSize;     ///< Expected file size in bytes
    };

    /** Returns the singleton instance. */
    static ecvTestDataRepository& instance();

    /** Get metadata for a dataset. */
    static DatasetInfo getDatasetInfo(Dataset kind);

    /** Returns the root data directory (~/cloudViewer_data). */
    static QString dataRoot();

    /** Returns the download directory (~/cloudViewer_data/download). */
    static QString downloadDir();

    /** Returns the extract directory (~/cloudViewer_data/extract). */
    static QString extractDir();

    /** Returns the path where a dataset zip should be stored. */
    static QString zipPath(Dataset kind);

    /** Returns the path where a dataset should be extracted. */
    static QString extractPath(Dataset kind);

    /** Returns true if the dataset is extracted or a valid zip is cached. */
    bool isDatasetAvailable(Dataset kind) const;

    /**
     * @brief Verify zip file integrity (size + MD5).
     * @param zipPath Path to the zip file
     * @param expectedMd5 Expected MD5 hash (empty = skip MD5 check)
     * @param expectedMinSize Minimum expected file size (0 = skip size check)
     * @return true if file exists, passes size check, and MD5 matches
     */
    static bool verifyZipIntegrity(const QString& zipPath,
                                   const QString& expectedMd5,
                                   qint64 expectedMinSize = 0);

    /**
     * @brief Start downloading a dataset.
     * Emits downloadProgress, downloadLogMessage, and downloadFinished signals.
     */
    void startDownload(Dataset kind);

    /** Cancel any in-progress download. */
    void cancelDownload();

    /** Returns true if a download is in progress. */
    bool isDownloadInProgress() const { return m_downloadInProgress; }

    /**
     * @brief Extract a downloaded zip to the extract directory.
     * Emits extractionProgress signal during extraction.
     * @param kind The dataset kind
     * @return true if extraction succeeded
     */
    bool extractDataset(Dataset kind);

    /**
     * @brief Extract a zip file to a target directory with callback progress.
     * This is a static utility method for callers that need callback-based
     * progress reporting (e.g., worker threads with weighted progress).
     * @param zipPath Path to the zip file
     * @param extractDir Target directory
     * @param onProgress Callback(current, total) for progress reporting
     * @return true if extraction succeeded
     */
    using ExtractProgressFn = std::function<void(int current, int total)>;
    static bool extractZip(const QString& zipPath,
                           const QString& extractDir,
                           const ExtractProgressFn& onProgress = {});

    /**
     * @brief Count entries in a zip file.
     * @param zipPath Path to the zip file
     * @return Number of entries, or 0 if unreadable
     */
    static int zipEntryCount(const QString& zipPath);

    /**
     * @brief Get list of image files from Monstree dataset.
     * @param bundleRoot Path to the extracted dataset root
     * @return Sorted list of absolute image file paths
     */
    static QStringList getMonstreeImages(const QString& bundleRoot);

    /**
     * @brief Find the first video file in FriendsFaces dataset.
     * @param bundleRoot Path to the extracted dataset root
     * @return Absolute path to the video file, or empty if not found
     */
    static QString findFriendsVideo(const QString& bundleRoot);

signals:
    /** Emitted during download with progress (0-100). */
    void downloadProgress(int percent, const QString& statusText);

    /** Emitted for log messages during download/extraction. */
    void downloadLogMessage(const QString& message);

    /** Emitted when download completes. success=false if download failed. */
    void downloadFinished(bool success, Dataset kind);

    /** Emitted during extraction with progress (current, total entries). */
    void extractionProgress(int current, int total);

    /** Emitted when extraction completes. success=false if extraction failed.
     */
    void extractionFinished(bool success, Dataset kind);

private slots:
    void onDownloaderProgress(qint64 received, qint64 total);
    void onDownloaderFinished(bool ok, const QString& destPath);

private:
    explicit ecvTestDataRepository(QObject* parent = nullptr);
    ~ecvTestDataRepository() override;

    // Prevent copying
    ecvTestDataRepository(const ecvTestDataRepository&) = delete;
    ecvTestDataRepository& operator=(const ecvTestDataRepository&) = delete;

    ecvModelDownloader* m_downloader = nullptr;
    bool m_downloadInProgress = false;
    Dataset m_currentDataset = Dataset::Monstree;
};
