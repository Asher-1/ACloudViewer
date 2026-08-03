// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceRegistryStore.h"

#include <QBuffer>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QThread>
#include <QUuid>
#include <cmath>
#include <cstring>
#include <shared_mutex>

namespace {

QString connectionNameForPath(const QString& path) {
    // Each thread needs its own QSQLITE connection (names are process-global).
    return QStringLiteral("face_registry_") + QString::number(qHash(path)) +
           QLatin1Char('_') +
           QString::number(
                   reinterpret_cast<quintptr>(QThread::currentThreadId()));
}

QByteArray embeddingToBlob(const std::vector<float>& v) {
    QByteArray bytes(reinterpret_cast<const char*>(v.data()),
                     static_cast<int>(v.size() * sizeof(float)));
    return bytes;
}

std::vector<float> embeddingFromBlob(const QByteArray& bytes) {
    std::vector<float> out(bytes.size() / static_cast<int>(sizeof(float)));
    if (!out.empty()) {
        std::memcpy(out.data(), bytes.constData(),
                    static_cast<size_t>(bytes.size()));
    }
    return out;
}

QByteArray imageToBlob(const QImage& img) {
    if (img.isNull()) return {};
    QByteArray bytes;
    QBuffer buf(&bytes);
    buf.open(QIODevice::WriteOnly);
    img.scaled(96, 96, Qt::KeepAspectRatio, Qt::SmoothTransformation)
            .save(&buf, "PNG");
    return bytes;
}

QImage imageFromBlob(const QByteArray& bytes) {
    if (bytes.isEmpty()) return {};
    QImage img;
    img.loadFromData(bytes, "PNG");
    return img;
}

}  // namespace

FaceRegistryStore::FaceRegistryStore(const QString& dbPath) : m_path(dbPath) {}

bool FaceRegistryStore::open() {
    std::unique_lock lock(m_mutex);
    return openUnlocked();
}

bool FaceRegistryStore::openUnlocked() {
    if (m_open) return true;
    QDir().mkpath(QFileInfo(m_path).absolutePath());

    const QString conn = connectionNameForPath(m_path);
    if (!QSqlDatabase::contains(conn)) {
        QSqlDatabase db =
                QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), conn);
        db.setDatabaseName(m_path);
        if (!db.open()) return false;
    } else if (!QSqlDatabase::database(conn).isOpen()) {
        if (!QSqlDatabase::database(conn).open()) return false;
    }

    migrateLegacyJson();
    if (!ensureSchema()) return false;
    if (!reloadFromDbUnlocked()) return false;
    m_open = true;
    return true;
}

void FaceRegistryStore::close() {
    std::unique_lock lock(m_mutex);
    const QString conn = connectionNameForPath(m_path);
    if (QSqlDatabase::contains(conn)) {
        QSqlDatabase::database(conn).close();
        QSqlDatabase::removeDatabase(conn);
    }
    m_entries.clear();
    m_cacheEmb.clear();
    m_cacheDim = 0;
    m_open = false;
}

void FaceRegistryStore::rebind(const QString& dbPath) {
    std::unique_lock lock(m_mutex);
    const QString oldConn = connectionNameForPath(m_path);
    if (QSqlDatabase::contains(oldConn)) {
        QSqlDatabase::database(oldConn).close();
        QSqlDatabase::removeDatabase(oldConn);
    }
    m_path = dbPath;
    m_entries.clear();
    m_cacheEmb.clear();
    m_cacheDim = 0;
    m_open = false;
    openUnlocked();
}

bool FaceRegistryStore::isOpen() const {
    std::shared_lock lock(m_mutex);
    return m_open;
}

std::vector<FaceRegistryEntry> FaceRegistryStore::entries() const {
    std::shared_lock lock(m_mutex);
    return m_entries;
}

bool FaceRegistryStore::ensureSchema() {
    QSqlQuery q(QSqlDatabase::database(connectionNameForPath(m_path)));
    return q.exec(
            QStringLiteral("CREATE TABLE IF NOT EXISTS faces ("
                           "  id TEXT PRIMARY KEY,"
                           "  name TEXT NOT NULL,"
                           "  model TEXT,"
                           "  dim INTEGER NOT NULL,"
                           "  created TEXT,"
                           "  embedding BLOB NOT NULL,"
                           "  thumb BLOB"
                           ")"));
}

bool FaceRegistryStore::migrateLegacyJson() const {
    const QString jsonPath = QFileInfo(m_path).absolutePath() +
                             QStringLiteral("/face_registry.json");
    if (!QFile::exists(jsonPath) || QFile::exists(m_path)) {
        return true;
    }

    QFile f(jsonPath);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
    if (!doc.isObject()) return false;

    QDir().mkpath(QFileInfo(m_path).absolutePath());
    const QString conn = connectionNameForPath(m_path);
    if (!QSqlDatabase::contains(conn)) {
        QSqlDatabase db =
                QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), conn);
        db.setDatabaseName(m_path);
        if (!db.open()) return false;
    }

    QSqlQuery q(QSqlDatabase::database(conn));
    q.exec(
            QStringLiteral("CREATE TABLE IF NOT EXISTS faces ("
                           "  id TEXT PRIMARY KEY,"
                           "  name TEXT NOT NULL,"
                           "  model TEXT,"
                           "  dim INTEGER NOT NULL,"
                           "  created TEXT,"
                           "  embedding BLOB NOT NULL,"
                           "  thumb BLOB"
                           ")"));

    const QJsonArray arr =
            doc.object().value(QStringLiteral("entries")).toArray();
    for (const QJsonValue& v : arr) {
        const QJsonObject o = v.toObject();
        const QString id = o.value(QStringLiteral("id")).toString();
        const QString name = o.value(QStringLiteral("name")).toString();
        if (id.isEmpty() || name.isEmpty()) continue;

        std::vector<float> emb;
        const QJsonArray embArr =
                o.value(QStringLiteral("embedding")).toArray();
        emb.reserve(static_cast<size_t>(embArr.size()));
        for (const QJsonValue& ev : embArr) {
            emb.push_back(static_cast<float>(ev.toDouble()));
        }
        if (emb.empty()) continue;

        QSqlQuery ins(QSqlDatabase::database(conn));
        ins.prepare(QStringLiteral(
                "INSERT OR IGNORE INTO "
                "faces(id,name,model,dim,created,embedding,thumb) "
                "VALUES(?,?,?,?,?,?,?)"));
        ins.addBindValue(id);
        ins.addBindValue(name);
        ins.addBindValue(o.value(QStringLiteral("model")).toString());
        ins.addBindValue(static_cast<int>(emb.size()));
        ins.addBindValue(o.value(QStringLiteral("created")).toString());
        ins.addBindValue(embeddingToBlob(emb));
        ins.addBindValue(QByteArray::fromBase64(
                o.value(QStringLiteral("thumb_b64")).toString().toLatin1()));
        ins.exec();
    }

    QFile::rename(jsonPath, jsonPath + QStringLiteral(".migrated"));
    return true;
}

bool FaceRegistryStore::reloadFromDb() {
    std::unique_lock lock(m_mutex);
    return reloadFromDbUnlocked();
}

bool FaceRegistryStore::reloadFromDbUnlocked() {
    m_entries.clear();
    QSqlQuery q(QSqlDatabase::database(connectionNameForPath(m_path)));
    if (!q.exec(QStringLiteral(
                "SELECT id,name,model,dim,created,embedding,thumb FROM faces "
                "ORDER BY created"))) {
        return false;
    }
    while (q.next()) {
        FaceRegistryEntry e;
        e.id = q.value(0).toString();
        e.name = q.value(1).toString();
        e.modelFile = q.value(2).toString();
        e.embedDim = q.value(3).toInt();
        e.createdUtc = q.value(4).toString();
        e.embedding = embeddingFromBlob(q.value(5).toByteArray());
        e.thumbnail = imageFromBlob(q.value(6).toByteArray());
        if (!e.id.isEmpty() && !e.embedding.empty()) {
            m_entries.push_back(std::move(e));
        }
    }
    rebuildCache();
    return true;
}

std::vector<float> FaceRegistryStore::normalizeEmbedding(
        const std::vector<float>& in) {
    double n = 0.0;
    for (float v : in) n += static_cast<double>(v) * v;
    if (n <= 0.0) return in;
    const float inv = static_cast<float>(1.0 / std::sqrt(n));
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = in[i] * inv;
    return out;
}

void FaceRegistryStore::rebuildCache() {
    m_cacheEmb.clear();
    m_cacheDim = 0;
    for (const FaceRegistryEntry& e : m_entries) {
        if (e.embedding.empty()) continue;
        if (m_cacheDim == 0) {
            m_cacheDim = static_cast<int>(e.embedding.size());
        }
        if (static_cast<int>(e.embedding.size()) != m_cacheDim) {
            continue;
        }
        const std::vector<float> norm = normalizeEmbedding(e.embedding);
        m_cacheEmb.insert(m_cacheEmb.end(), norm.begin(), norm.end());
    }
}

bool FaceRegistryStore::addEntry(FaceRegistryEntry entry) {
    std::unique_lock lock(m_mutex);
    if (!m_open && !openUnlocked()) return false;
    if (entry.id.isEmpty()) {
        entry.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
    }
    if (entry.createdUtc.isEmpty()) {
        entry.createdUtc =
                QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    }
    if (entry.embedDim <= 0) {
        entry.embedDim = static_cast<int>(entry.embedding.size());
    }

    QSqlQuery q(QSqlDatabase::database(connectionNameForPath(m_path)));
    q.prepare(
            QStringLiteral("INSERT OR REPLACE INTO "
                           "faces(id,name,model,dim,created,embedding,thumb) "
                           "VALUES(?,?,?,?,?,?,?)"));
    q.addBindValue(entry.id);
    q.addBindValue(entry.name);
    q.addBindValue(entry.modelFile);
    q.addBindValue(entry.embedDim);
    q.addBindValue(entry.createdUtc);
    q.addBindValue(embeddingToBlob(entry.embedding));
    q.addBindValue(imageToBlob(entry.thumbnail));
    if (!q.exec()) return false;
    return reloadFromDbUnlocked();
}

bool FaceRegistryStore::updateEntry(const QString& id, const QString& name) {
    std::unique_lock lock(m_mutex);
    if (!m_open && !openUnlocked()) return false;
    QSqlQuery q(QSqlDatabase::database(connectionNameForPath(m_path)));
    q.prepare(QStringLiteral("UPDATE faces SET name=? WHERE id=?"));
    q.addBindValue(name);
    q.addBindValue(id);
    if (!q.exec()) return false;
    return reloadFromDbUnlocked();
}

bool FaceRegistryStore::removeEntry(const QString& id) {
    std::unique_lock lock(m_mutex);
    if (!m_open && !openUnlocked()) return false;
    QSqlQuery q(QSqlDatabase::database(connectionNameForPath(m_path)));
    q.prepare(QStringLiteral("DELETE FROM faces WHERE id=?"));
    q.addBindValue(id);
    if (!q.exec()) return false;
    return reloadFromDbUnlocked();
}

void FaceRegistryStore::clear() {
    std::unique_lock lock(m_mutex);
    if (!m_open && !openUnlocked()) return;
    QSqlQuery q(QSqlDatabase::database(connectionNameForPath(m_path)));
    q.exec(QStringLiteral("DELETE FROM faces"));
    reloadFromDbUnlocked();
}

float FaceRegistryStore::cosineDistance(const std::vector<float>& a,
                                        const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1.0f;
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 1.0f;
    return static_cast<float>(1.0 - dot / (std::sqrt(na) * std::sqrt(nb)));
}

std::optional<FaceAuthMatch> FaceRegistryStore::bestMatch(
        const std::vector<float>& query, float maxDistance) const {
    std::shared_lock lock(m_mutex);
    if (query.empty() || m_cacheDim <= 0 ||
        static_cast<int>(query.size()) != m_cacheDim ||
        m_cacheEmb.size() < static_cast<size_t>(m_cacheDim)) {
        return std::nullopt;
    }

    const std::vector<float> qn = normalizeEmbedding(query);
    std::optional<FaceAuthMatch> best;
    size_t cachedRow = 0;
    for (const FaceRegistryEntry& e : m_entries) {
        if (static_cast<int>(e.embedding.size()) != m_cacheDim) {
            continue;
        }
        const float* row_emb =
                m_cacheEmb.data() + cachedRow * static_cast<size_t>(m_cacheDim);
        double dot = 0.0;
        for (int i = 0; i < m_cacheDim; ++i) {
            dot += static_cast<double>(qn[static_cast<size_t>(i)]) * row_emb[i];
        }
        const float dist = static_cast<float>(1.0 - dot);
        if (dist <= maxDistance && (!best || dist < best->distance)) {
            best = FaceAuthMatch{e, dist};
        }
        ++cachedRow;
    }
    return best;
}

std::optional<FaceAuthMatch> FaceRegistryStore::nearestMatch(
        const std::vector<float>& query) const {
    std::shared_lock lock(m_mutex);
    if (query.empty() || m_cacheDim <= 0 ||
        static_cast<int>(query.size()) != m_cacheDim ||
        m_cacheEmb.size() < static_cast<size_t>(m_cacheDim)) {
        return std::nullopt;
    }

    const std::vector<float> qn = normalizeEmbedding(query);
    std::optional<FaceAuthMatch> best;
    size_t cachedRow = 0;
    for (const FaceRegistryEntry& e : m_entries) {
        if (static_cast<int>(e.embedding.size()) != m_cacheDim) {
            continue;
        }
        const float* row_emb =
                m_cacheEmb.data() + cachedRow * static_cast<size_t>(m_cacheDim);
        double dot = 0.0;
        for (int i = 0; i < m_cacheDim; ++i) {
            dot += static_cast<double>(qn[static_cast<size_t>(i)]) * row_emb[i];
        }
        const float dist = static_cast<float>(1.0 - dot);
        if (!best || dist < best->distance) {
            best = FaceAuthMatch{e, dist};
        }
        ++cachedRow;
    }
    return best;
}
