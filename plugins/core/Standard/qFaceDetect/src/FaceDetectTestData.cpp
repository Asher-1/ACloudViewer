// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceDetectTestData.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImageReader>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSet>
#include <QSettings>

#include "FaceRegistryStore.h"

namespace FaceDetectTestData {

namespace {

QString absPathIfExists(const QString& path) {
    return QFileInfo::exists(path) ? QFileInfo(path).absoluteFilePath()
                                   : QString();
}

QString findByFileNames(const QString& root, const QStringList& names) {
    for (const QString& name : names) {
        QDirIterator it(root, QStringList{name}, QDir::Files,
                        QDirIterator::Subdirectories);
        if (it.hasNext()) return absPathIfExists(it.next());
    }
    return {};
}

QString findFirstVideo(const QString& root) {
    const QStringList patterns = {
            QStringLiteral("*.mp4"), QStringLiteral("*.mkv"),
            QStringLiteral("*.avi"), QStringLiteral("*.mov"),
            QStringLiteral("*.webm")};
    QDirIterator it(root, patterns, QDir::Files, QDirIterator::Subdirectories);
    QString best;
    while (it.hasNext()) {
        const QString path = it.next();
        const QString base = QFileInfo(path).fileName().toLower();
        if (base.contains(QStringLiteral("friend")))
            return absPathIfExists(path);
        if (best.isEmpty()) best = absPathIfExists(path);
    }
    return best;
}

QStringList listImages(const QString& root) {
    const QStringList patterns = {
            QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"), QStringLiteral("*.webp")};
    QStringList out;
    QDirIterator it(root, patterns, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) out.append(it.next());
    out.sort(Qt::CaseInsensitive);
    return out;
}

void applyManifest(const QJsonObject& obj,
                   const QString& root,
                   FaceDetectFriendsBundle* out) {
    auto relPath = [&](const char* key) -> QString {
        const QString rel = obj.value(QString::fromLatin1(key)).toString();
        if (rel.isEmpty()) return {};
        return absPathIfExists(QDir(root).filePath(rel));
    };
    out->videoPath = relPath("video");
    out->registerImage = relPath("register_image");
    out->authProbeImage = relPath("auth_probe_image");
    out->batchImage = relPath("batch_image");
    out->registryDbPath = relPath("registry_db");
    out->registerName = obj.value(QStringLiteral("register_name")).toString();
}

void fillHeuristics(const QString& root, FaceDetectFriendsBundle* out) {
    auto knownRel = [&](const QString& rel) -> QString {
        return absPathIfExists(QDir(root).filePath(rel));
    };

    // Layout of FriendsFaces release (no manifest.json in zip).
    if (out->videoPath.isEmpty()) {
        out->videoPath = knownRel(QStringLiteral("query/friends_demo.mp4"));
    }
    if (out->registerImage.isEmpty()) {
        out->registerImage = knownRel(QStringLiteral("query/Rachel.png"));
    }
    if (out->authProbeImage.isEmpty()) {
        out->authProbeImage = knownRel(QStringLiteral("query/friends1.jpg"));
        if (out->authProbeImage.isEmpty()) {
            out->authProbeImage =
                    knownRel(QStringLiteral("query/multiple_people.jpg"));
        }
    }
    if (out->batchImage.isEmpty()) {
        out->batchImage = knownRel(QStringLiteral("output/friends1.jpg"));
    }
    if (out->batchImage.isEmpty()) {
        out->batchImage = knownRel(QStringLiteral("query/friends1.jpg"));
    }
    if (out->batchImage.isEmpty()) {
        out->batchImage = knownRel(QStringLiteral("query/multiple_people.jpg"));
    }
    if (out->registerName.isEmpty()) {
        out->registerName = QStringLiteral("Rachel");
    }
    if (out->registryDbPath.isEmpty()) {
        out->registryDbPath =
                registryPathForModel(root, QStringLiteral("buffalo_l.gguf"));
    }

    if (out->videoPath.isEmpty()) {
        out->videoPath =
                findByFileNames(root, {QStringLiteral("friends_demo.mp4"),
                                       QStringLiteral("Friends.mp4")});
    }
    if (out->videoPath.isEmpty()) out->videoPath = findFirstVideo(root);

    const QStringList images = listImages(root);
    auto pick = [&](const QStringList& tokens) -> QString {
        for (const QString& path : images) {
            const QString lower = path.toLower();
            for (const QString& tok : tokens) {
                if (lower.contains(tok)) return absPathIfExists(path);
            }
        }
        return {};
    };

    if (out->registerImage.isEmpty()) {
        out->registerImage =
                pick({QStringLiteral("register"), QStringLiteral("enroll"),
                      QStringLiteral("rachel")});
    }
    if (out->authProbeImage.isEmpty()) {
        out->authProbeImage =
                pick({QStringLiteral("probe"), QStringLiteral("auth"),
                      QStringLiteral("verify")});
    }
    if (out->batchImage.isEmpty()) {
        out->batchImage =
                pick({QStringLiteral("friends1"), QStringLiteral("multiple"),
                      QStringLiteral("group"), QStringLiteral("together"),
                      QStringLiteral("batch"), QStringLiteral("detect")});
    }
    // Do not fall back to arbitrary single portraits — keep group-shot intent.
    if (out->batchImage.isEmpty() && !out->authProbeImage.isEmpty()) {
        out->batchImage = out->authProbeImage;
    }

    if (out->registerImage.isEmpty() && !images.isEmpty()) {
        out->registerImage =
                pick({QStringLiteral("rachel"), QStringLiteral("register"),
                      QStringLiteral("enroll")});
        if (out->registerImage.isEmpty()) {
            out->registerImage = absPathIfExists(images.front());
        }
    }
    if (out->authProbeImage.isEmpty() && images.size() > 1) {
        for (const QString& path : images) {
            if (path != out->registerImage) {
                out->authProbeImage = absPathIfExists(path);
                break;
            }
        }
    }

    if (out->registerName.isEmpty()) {
        if (!out->registerImage.isEmpty()) {
            out->registerName =
                    QFileInfo(out->registerImage).completeBaseName();
        } else {
            out->registerName = QStringLiteral("Rachel");
        }
    }
}

QString locateBundleRoot() {
    // Only return the exact FriendsFaces extract path.
    // Do NOT fall back to scanning other directories under extract/,
    // as that could incorrectly match objects_detection_data (which
    // also contains video+images) and fill the bundle with wrong paths.
    const QString primary = ecvTestDataRepository::extractPath(
            ecvTestDataRepository::Dataset::FriendsFaces);
    if (QDir(primary).exists()) return primary;

    return {};
}

}  // namespace

// Path and verification functions are now inline in the header,
// delegating to ecvTestDataRepository for cross-plugin consistency.

// zipEntryCount and extractZip delegate to ecvTestDataRepository
// to avoid duplicating minizip extraction logic.
int zipEntryCount(const QString& zipPath) {
    return ecvTestDataRepository::zipEntryCount(zipPath);
}

bool extractZip(const QString& zipPath,
                const QString& extractParentDir,
                const ExtractProgressFn& onProgress) {
    return ecvTestDataRepository::extractZip(zipPath, extractParentDir,
                                             onProgress);
}

QVector<FaceDetectGalleryEntry> loadGalleryEntries(const QString& bundleRoot) {
    QVector<FaceDetectGalleryEntry> out;
    const QString galleryDir =
            QDir(bundleRoot).filePath(QStringLiteral("gallery"));
    const QString labelPath =
            QDir(galleryDir).filePath(QStringLiteral("label.txt"));
    QFile labelFile(labelPath);
    if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) return out;

    while (!labelFile.atEnd()) {
        const QString line = QString::fromUtf8(labelFile.readLine()).trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) continue;

        const int tab = line.indexOf(QLatin1Char('\t'));
        if (tab <= 0) continue;

        QString rel = line.left(tab).trimmed();
        const QString name = line.mid(tab + 1).trimmed();
        if (name.isEmpty()) continue;

        if (rel.startsWith(QStringLiteral("./"))) rel = rel.mid(2);
        if (rel.startsWith(QLatin1Char('/'))) rel = rel.mid(1);

        const QString imagePath =
                absPathIfExists(QDir(galleryDir).filePath(rel));
        if (imagePath.isEmpty()) continue;

        out.append(FaceDetectGalleryEntry{name, imagePath});
    }
    return out;
}

QVector<FaceDetectGalleryEntry> queryPortraitEntries(
        const QString& bundleRoot) {
    QVector<FaceDetectGalleryEntry> out;
    const QString queryDir = QDir(bundleRoot).filePath(QStringLiteral("query"));
    const QStringList preferred = {
            QStringLiteral("Rachel.png"),   QStringLiteral("Monica.png"),
            QStringLiteral("Phoebe.png"),   QStringLiteral("Joey.png"),
            QStringLiteral("Chandler.png"), QStringLiteral("Ross.png"),
            QStringLiteral("friends1.jpg")};

    for (const QString& filename : preferred) {
        const QString path = absPathIfExists(QDir(queryDir).filePath(filename));
        if (path.isEmpty()) continue;
        const QString name = QFileInfo(path).completeBaseName();
        out.append(FaceDetectGalleryEntry{name, path});
        if (out.size() >= 6) break;
    }

    if (out.size() >= 6) return out;

    const QVector<FaceDetectGalleryEntry> gallery =
            loadGalleryEntries(bundleRoot);
    QSet<QString> seenNames;
    for (const FaceDetectGalleryEntry& entry : gallery) {
        if (seenNames.contains(entry.name)) continue;
        seenNames.insert(entry.name);
        out.append(entry);
        if (out.size() >= 6) break;
    }
    return out;
}

QString largestImageInDir(const QString& dirPath) {
    QDir dir(dirPath);
    if (!dir.exists()) return {};
    const QStringList filters = {
            QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"), QStringLiteral("*.webp")};
    QString best;
    qint64 bestArea = 0;
    for (const QString& fn : dir.entryList(filters, QDir::Files)) {
        const QString path = dir.filePath(fn);
        QImageReader reader(path);
        reader.setAutoTransform(true);
        const QSize size = reader.size();
        if (!size.isValid()) continue;
        const qint64 area = qint64(size.width()) * size.height();
        if (area > bestArea) {
            bestArea = area;
            best = path;
        }
    }
    return best;
}

QString queryPortraitPath(const QString& bundleRoot, const QString& name) {
    const QString queryDir = QDir(bundleRoot).filePath(QStringLiteral("query"));
    static const QStringList kExts = {
            QStringLiteral(".png"), QStringLiteral(".jpg"),
            QStringLiteral(".jpeg"), QStringLiteral(".webp")};
    for (const QString& ext : kExts) {
        const QString path =
                absPathIfExists(QDir(queryDir).filePath(name + ext));
        if (!path.isEmpty()) return path;
    }
    return {};
}

QString fixedRegistrationImagePath(const QString& bundleRoot,
                                   const QString& name) {
    static const QHash<QString, QString> kFixedGalleryFrontals = {
            {QStringLiteral("Chandler"), QStringLiteral("Chandler00009.png")},
            {QStringLiteral("Monica"), QStringLiteral("Monica00022.png")},
            {QStringLiteral("Phoebe"), QStringLiteral("Phoebe00001.jpg")},
            {QStringLiteral("Rachel"), QStringLiteral("Rachel00002.png")},
            {QStringLiteral("Joey"), QStringLiteral("Joey00030.jpg")},
            {QStringLiteral("Ross"), QStringLiteral("Ross00002.png")},
    };
    const auto it = kFixedGalleryFrontals.constFind(name);
    if (it == kFixedGalleryFrontals.constEnd()) return {};
    const QString galleryDir =
            QDir(bundleRoot).filePath(QStringLiteral("gallery/%1").arg(name));
    return absPathIfExists(QDir(galleryDir).filePath(it.value()));
}

QVector<FaceDetectGalleryEntry> registrationEntriesForBundle(
        const QString& bundleRoot) {
    static const QStringList kCast = {
            QStringLiteral("Chandler"), QStringLiteral("Monica"),
            QStringLiteral("Phoebe"),   QStringLiteral("Rachel"),
            QStringLiteral("Joey"),     QStringLiteral("Ross")};
    QVector<FaceDetectGalleryEntry> out;
    for (const QString& name : kCast) {
        QString path = fixedRegistrationImagePath(bundleRoot, name);
        if (path.isEmpty()) {
            path = queryPortraitPath(bundleRoot, name);
        }
        if (path.isEmpty()) {
            const QString galleryRoot =
                    QDir(bundleRoot).filePath(QStringLiteral("gallery"));
            path = largestImageInDir(QDir(galleryRoot).filePath(name));
        }
        if (path.isEmpty()) continue;
        out.append(FaceDetectGalleryEntry{name, path});
    }
    return out;
}

QString registryPathForModel(const QString& bundleRoot,
                             const QString& modelFilename) {
    const QString stem = QFileInfo(modelFilename).completeBaseName();
    const QString safeStem = stem.isEmpty() ? QStringLiteral("default") : stem;
    return QDir(bundleRoot)
            .filePath(QStringLiteral("face_registry_%1.db").arg(safeStem));
}

bool resolveBundle(FaceDetectFriendsBundle* out) {
    if (!out) return false;
    *out = FaceDetectFriendsBundle{};
    const QString root = locateBundleRoot();
    if (root.isEmpty()) return false;
    out->extractRoot = root;

    const QString manifestPath =
            QDir(root).filePath(QStringLiteral("manifest.json"));
    QFile mf(manifestPath);
    if (mf.open(QIODevice::ReadOnly)) {
        const QJsonDocument doc = QJsonDocument::fromJson(mf.readAll());
        if (doc.isObject()) applyManifest(doc.object(), root, out);
    }

    fillHeuristics(root, out);
    if (out->galleryEntries.isEmpty()) {
        out->galleryEntries = registrationEntriesForBundle(root);
        if (out->galleryEntries.isEmpty()) {
            out->galleryEntries = loadGalleryEntries(root);
        }
    }
    if (out->registryDbPath.isEmpty()) {
        out->registryDbPath =
                registryPathForModel(root, QStringLiteral("buffalo_l.gguf"));
    }
    return out->isUsableForLive() || out->isUsableForRegistry();
}

bool isFriendsBundlePath(const QString& path) {
    if (path.isEmpty()) return false;
    const QString norm = QDir::cleanPath(QDir::fromNativeSeparators(path));
    if (norm.contains(QStringLiteral("/friends_faces/"), Qt::CaseInsensitive) ||
        norm.endsWith(QStringLiteral("/friends_faces"), Qt::CaseInsensitive)) {
        return true;
    }
    FaceDetectFriendsBundle bundle;
    if (!resolveBundle(&bundle) || bundle.extractRoot.isEmpty()) {
        return false;
    }
    const QString root = QDir::cleanPath(bundle.extractRoot);
    return norm == root || norm.startsWith(root + QLatin1Char('/'));
}

QString activeRegistrySettingsKey() {
    return QStringLiteral("qFaceDetect/activeRegistryDbPath");
}

QString manualLiveVideoSettingsKey() {
    return QStringLiteral("qFaceDetect/manualLiveVideoPath");
}

QString manualBatchImageSettingsKey() {
    return QStringLiteral("qFaceDetect/manualBatchImagePath");
}

QString manualRegistryDbSettingsKey() {
    return QStringLiteral("qFaceDetect/manualRegistryDbPath");
}

void purgeFriendsPathsFromSettings() {
    QSettings settings;
    const auto purgeKey = [&settings](const char* key) {
        const QString value =
                settings.value(QString::fromLatin1(key)).toString();
        if (isFriendsBundlePath(value)) {
            settings.remove(QString::fromLatin1(key));
        }
    };
    purgeKey("qFaceDetect/batchImagePath");
    purgeKey("qFaceDetect/liveVideoPath");
    purgeKey("qFaceDetect/liveRegistryDbPath");
    purgeKey("qFaceDetect/registryDbPath");
    purgeKey("qFaceDetect/activeRegistryDbPath");
    // Drop auto-saved friends paths from manual keys if they leaked in.
    purgeKey("qFaceDetect/manualLiveVideoPath");
    purgeKey("qFaceDetect/manualBatchImagePath");
    purgeKey("qFaceDetect/manualRegistryDbPath");
}

int registryEntryCount(const QString& dbPath) {
    if (dbPath.isEmpty() || !QFileInfo::exists(dbPath)) return 0;
    FaceRegistryStore store(dbPath);
    if (!store.open()) return 0;
    return static_cast<int>(store.entries().size());
}

QString discoverRegistryDbPath(const QString& modelFilename) {
    const auto pickBest = [&](const QString& path) -> QString {
        return registryEntryCount(path) > 0 ? path : QString{};
    };

    FaceDetectFriendsBundle bundle;
    if (resolveBundle(&bundle) && !bundle.extractRoot.isEmpty()) {
        const QString primary =
                registryPathForModel(bundle.extractRoot, modelFilename);
        if (const QString found = pickBest(primary); !found.isEmpty()) {
            return found;
        }

        QDir dir(bundle.extractRoot);
        const QStringList candidates = dir.entryList(
                {QStringLiteral("face_registry_*.db")}, QDir::Files);
        for (const QString& fn : candidates) {
            if (const QString found = pickBest(dir.filePath(fn));
                !found.isEmpty()) {
                return found;
            }
        }
    }

    // Broader scan: any registry DB under cloudViewer_data/extract (not only
    // friends_faces bundle root).
    const QString extractRoot = ecvTestDataRepository::extractDir();
    QString bestMatch;
    int bestCount = 0;
    QDirIterator it(extractRoot, {QStringLiteral("face_registry_*.db")},
                    QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString path = it.next();
        const int count = registryEntryCount(path);
        if (count <= 0) continue;
        const QString stemBase =
                QFileInfo(modelFilename).completeBaseName().toLower();
        if (!stemBase.isEmpty() && path.toLower().contains(stemBase)) {
            return path;
        }
        if (count > bestCount) {
            bestCount = count;
            bestMatch = path;
        }
    }
    return bestMatch;
}

QString groupPhotoPath(const FaceDetectFriendsBundle& bundle) {
    if (!bundle.authProbeImage.isEmpty()) return bundle.authProbeImage;
    if (!bundle.batchImage.isEmpty()) return bundle.batchImage;
    if (!bundle.extractRoot.isEmpty()) {
        const QString friends1 =
                QDir(bundle.extractRoot)
                        .filePath(QStringLiteral("query/friends1.jpg"));
        if (QFileInfo::exists(friends1))
            return QFileInfo(friends1).absoluteFilePath();
        const QString outputFriends1 =
                QDir(bundle.extractRoot)
                        .filePath(QStringLiteral("output/friends1.jpg"));
        if (QFileInfo::exists(outputFriends1)) {
            return QFileInfo(outputFriends1).absoluteFilePath();
        }
    }
    return {};
}

bool verifyTestImagePair(const FaceDetectFriendsBundle& bundle,
                         QString* imageA,
                         QString* imageB) {
    if (!imageA || !imageB) return false;
    imageA->clear();
    imageB->clear();

    // Same-person verify demo: query portrait vs gallery frontal of
    // registerName.
    if (!bundle.registerImage.isEmpty() && !bundle.registerName.isEmpty() &&
        !bundle.extractRoot.isEmpty()) {
        *imageA = bundle.registerImage;
        const QString gallery = fixedRegistrationImagePath(bundle.extractRoot,
                                                           bundle.registerName);
        if (!gallery.isEmpty() && gallery != *imageA) {
            *imageB = gallery;
            return true;
        }
    }

    if (!bundle.registerImage.isEmpty()) {
        *imageA = bundle.registerImage;
    }
    for (const FaceDetectGalleryEntry& entry : bundle.galleryEntries) {
        if (entry.imagePath.isEmpty()) continue;
        if (imageA->isEmpty()) {
            *imageA = entry.imagePath;
            continue;
        }
        if (entry.imagePath != *imageA) {
            *imageB = entry.imagePath;
            break;
        }
    }
    if (imageB->isEmpty() && !bundle.galleryEntries.isEmpty()) {
        for (const FaceDetectGalleryEntry& entry : bundle.galleryEntries) {
            if (!entry.imagePath.isEmpty() && entry.imagePath != *imageA) {
                *imageB = entry.imagePath;
                break;
            }
        }
    }
    if (imageB->isEmpty() && !bundle.registerImage.isEmpty()) {
        const QDir galleryDir =
                QFileInfo(bundle.registerImage).absoluteDir().absolutePath();
        const QStringList filters = {
                QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
                QStringLiteral("*.png"), QStringLiteral("*.webp")};
        for (const QString& fn :
             QDir(galleryDir).entryList(filters, QDir::Files)) {
            const QString path = QDir(galleryDir).filePath(fn);
            if (path != *imageA) {
                *imageB = path;
                break;
            }
        }
    }
    return !imageA->isEmpty() && !imageB->isEmpty();
}

}  // namespace FaceDetectTestData
