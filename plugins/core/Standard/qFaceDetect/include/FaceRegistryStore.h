// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QString>
#include <optional>
#include <shared_mutex>
#include <vector>

/** SQLite-backed face registry with in-memory embedding cache for fast auth. */
struct FaceRegistryEntry {
    QString id;
    QString name;
    QString modelFile;
    int embedDim = 0;
    std::vector<float> embedding;
    QString createdUtc;
    QImage thumbnail;
};

struct FaceAuthMatch {
    FaceRegistryEntry entry;
    float distance = 1.0f;
};

class FaceRegistryStore {
public:
    explicit FaceRegistryStore(const QString& dbPath);

    bool open();
    /** Close and reopen on a new SQLite path (same object, thread-safe). */
    void rebind(const QString& dbPath);
    void close();
    bool isOpen() const;

    /** Thread-safe snapshot (typical gallery size is small). */
    std::vector<FaceRegistryEntry> entries() const;

    bool addEntry(FaceRegistryEntry entry);
    bool updateEntry(const QString& id, const QString& name);
    bool removeEntry(const QString& id);
    void clear();

    /** Lowest cosine distance below threshold wins; uses normalized cache. */
    std::optional<FaceAuthMatch> bestMatch(const std::vector<float>& query,
                                           float maxDistance) const;
    /** Closest enrolled identity regardless of threshold (for diagnostics). */
    std::optional<FaceAuthMatch> nearestMatch(const std::vector<float>& query) const;

    static float cosineDistance(const std::vector<float>& a,
                                const std::vector<float>& b);

    QString path() const { return m_path; }

private:
    bool ensureSchema();
    /** Caller must hold m_mutex (unique). */
    bool openUnlocked();
    bool reloadFromDb();
    bool reloadFromDbUnlocked();
    void rebuildCache();
    static std::vector<float> normalizeEmbedding(const std::vector<float>& in);
    bool migrateLegacyJson() const;

    QString m_path;
    std::vector<FaceRegistryEntry> m_entries;
    std::vector<float> m_cacheEmb;  // row-major N x D, L2-normalized
    int m_cacheDim = 0;
    bool m_open = false;
    mutable std::shared_mutex m_mutex;
};
