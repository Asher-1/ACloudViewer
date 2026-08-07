// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceDetectTestDataWorker.h"

#include <QDir>
#include <QFileInfo>
#include <QImageReader>
#include <QSet>
#include <algorithm>

#include "FaceDetectEmbedHelpers.h"
#include "FaceDetectModelContext.h"
#include "FaceDetectTestData.h"

#ifdef AICore_ENABLED
#include "aicore/facedetect_capi.h"
#endif

namespace {

QStringList imagesInPersonDir(const QString& personDir) {
    QDir dir(personDir);
    if (!dir.exists()) return {};
    const QStringList filters = {
            QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"), QStringLiteral("*.webp")};
    QStringList paths;
    for (const QString& fn : dir.entryList(filters, QDir::Files)) {
        paths << dir.filePath(fn);
    }
    std::sort(paths.begin(), paths.end(),
              [](const QString& a, const QString& b) {
                  QImageReader ra(a);
                  QImageReader rb(b);
                  ra.setAutoTransform(true);
                  rb.setAutoTransform(true);
                  const QSize sa = ra.size();
                  const QSize sb = rb.size();
                  const qint64 areaA =
                          sa.isValid() ? qint64(sa.width()) * sa.height() : 0;
                  const qint64 areaB =
                          sb.isValid() ? qint64(sb.width()) * sb.height() : 0;
                  return areaA > areaB;
              });
    return paths;
}

#ifdef AICore_ENABLED
QStringList registrationCandidatesForName(const QString& bundleRoot,
                                          const QString& name) {
    QStringList out;
    QSet<QString> seen;
    auto add = [&](const QString& path) {
        if (path.isEmpty() || seen.contains(path)) return;
        seen.insert(path);
        out.append(path);
    };
    add(FaceDetectTestData::fixedRegistrationImagePath(bundleRoot, name));
    add(FaceDetectTestData::queryPortraitPath(bundleRoot, name));
    const QString personDir =
            QDir(bundleRoot).filePath(QStringLiteral("gallery/%1").arg(name));
    for (const QString& path : imagesInPersonDir(personDir)) {
        add(path);
    }
    return out;
}
#endif

}  // namespace

namespace {

// Relative cost units: embed/detect dominate; zip I/O is cheap.
constexpr int kWeightExtractEntry = 1;
constexpr int kWeightResolveBundle = 1;
constexpr int kWeightModelLoad = 3;
constexpr int kWeightRegisterEmbed = 10;
constexpr int kWeightVerifyDetect = 8;
constexpr int kWeightVerifyEmbed = 10;
/** Conservative face count for friends1 group photo before detect runs. */
constexpr int kVerifyFaceEstimate = 6;

class WeightedProgress {
public:
    using Reporter =
            std::function<void(int completed, int total, const QString& label)>;

    explicit WeightedProgress(Reporter reporter)
        : m_report(std::move(reporter)) {}

    void setTotal(int total) { m_total = std::max(1, total); }

    void report(int completed, const QString& label) const {
        if (!m_report) return;
        const int clamped = std::max(0, std::min(completed, m_total));
        m_report(clamped, m_total, label);
    }

    void advance(int weight, const QString& label) {
        m_completed += weight;
        report(m_completed, label);
    }

    void setCompleted(int completed, const QString& label) {
        m_completed = completed;
        report(m_completed, label);
    }

    int completed() const { return m_completed; }
    int total() const { return m_total; }

    void bumpTotal(int extra) { m_total += std::max(0, extra); }

    void adjustTotal(int delta) {
        m_total = std::max(m_completed, m_total + delta);
    }

private:
    Reporter m_report;
    int m_completed = 0;
    int m_total = 1;
};

QVector<FaceDetectGalleryEntry> resolveRegistrationEntries(
        const FaceDetectTestDataWorker::Job& job) {
    if (!job.bundle.galleryEntries.isEmpty()) {
        return job.bundle.galleryEntries;
    }
    if (!job.bundle.extractRoot.isEmpty()) {
        return FaceDetectTestData::registrationEntriesForBundle(
                job.bundle.extractRoot);
    }
    return {};
}

int estimateTotalWeight(const FaceDetectTestDataWorker::Job& job,
                        int extractEntries,
                        int registerCount) {
    int total = 0;
    if (job.extractZipFirst) {
        const int files = extractEntries > 0 ? extractEntries : 1;
        total += files * kWeightExtractEntry + kWeightResolveBundle;
    }
    if (job.registerGallery || job.runVerify) {
        total += kWeightModelLoad;
    }
    if (job.registerGallery) {
        total += registerCount * kWeightRegisterEmbed;
    }
    if (job.runVerify) {
        total += kWeightVerifyDetect + kVerifyFaceEstimate * kWeightVerifyEmbed;
    }
    return std::max(1, total);
}

}  // namespace

FaceDetectTestDataWorker::FaceDetectTestDataWorker(QObject* parent)
    : QThread(parent) {}

void FaceDetectTestDataWorker::setJob(Job job) { m_job = std::move(job); }

void FaceDetectTestDataWorker::run() {
#ifndef AICore_ENABLED
    emit finished(false, 0, 0, 0);
    return;
#else
    Job job = m_job;

    const int extractEntries =
            job.extractZipFirst ? FaceDetectTestData::zipEntryCount(job.zipPath)
                                : 0;
    const QVector<FaceDetectGalleryEntry> plannedEntries =
            resolveRegistrationEntries(job);
    const int registerCountEstimate =
            job.registerGallery ? plannedEntries.size() : 0;

    WeightedProgress progress(
            [this](int completed, int total, const QString& label) {
                emit phaseProgress(completed, total, label);
            });
    progress.setTotal(
            estimateTotalWeight(job, extractEntries, registerCountEstimate));

    if (job.extractZipFirst) {
        const int extractBase = progress.completed();
        int extractFileTotal = extractEntries;
        const int extractWeight =
                (extractFileTotal > 0 ? extractFileTotal : 1) *
                kWeightExtractEntry;
        if (!FaceDetectTestData::extractZip(
                    job.zipPath, job.extractParentDir,
                    [&](int current, int total) {
                        if (extractFileTotal <= 0 && total > 0) {
                            extractFileTotal = total;
                        }
                        const int weight =
                                (extractFileTotal > 0 ? extractFileTotal
                                                      : total) *
                                kWeightExtractEntry;
                        const int done = extractBase +
                                         weight * current / std::max(1, total);
                        progress.report(done,
                                        tr("Extracting archive (%1/%2 files)…")
                                                .arg(current)
                                                .arg(total));
                    })) {
            emit logMessage(tr("[Test data] Failed to extract archive."));
            emit finished(false, 0, 0, 0);
            return;
        }
        progress.setCompleted(
                extractBase + extractWeight,
                tr("Extract complete (%1 files).")
                        .arg(extractFileTotal > 0 ? extractFileTotal
                                                  : extractEntries));

        progress.report(progress.completed(),
                        tr("Resolving FriendsFaces bundle…"));
        FaceDetectFriendsBundle resolved;
        if (!FaceDetectTestData::resolveBundle(&resolved)) {
            emit logMessage(tr("[Test data] Extracted but bundle not found."));
            emit finished(false, 0, 0, 0);
            return;
        }
        job.bundle = resolved;
        progress.advance(kWeightResolveBundle, tr("Bundle ready."));
    }

    const QVector<FaceDetectGalleryEntry> registrationEntries =
            resolveRegistrationEntries(job);
    const int registerCount =
            job.registerGallery ? registrationEntries.size() : 0;
    progress.setTotal(estimateTotalWeight(job, extractEntries, registerCount));

    if (!job.registerGallery && !job.runVerify) {
        progress.report(progress.total(), tr("Test data ready."));
        emit finished(true, 0, 0, 0);
        return;
    }

    if (job.modelPath.isEmpty() || !QFileInfo::exists(job.modelPath)) {
        emit logMessage(
                tr("[Test data] Face model not available for registration."));
        emit finished(false, 0, 0, 0);
        return;
    }

    progress.report(progress.completed(), tr("Loading face model…"));
    FaceDetectInferenceGuard inferenceGuard(job.device);
    FaceDetectModelContext modelCtx;
    if (!modelCtx.load(job.modelPath, job.device, job.threads)) {
        emit logMessage(tr("[Test data] Failed to load face model."));
        emit finished(false, 0, 0, 0);
        return;
    }
    progress.advance(kWeightModelLoad, tr("Face model loaded."));
    aicore_facedetect_ctx* ctx = modelCtx.get();

    FaceRegistryStore store(job.registryPath);
    if (!store.open()) {
        emit logMessage(tr("[Test data] Failed to open registry: %1")
                                .arg(job.registryPath));
        emit finished(false, 0, 0, 0);
        return;
    }

    int registered = 0;
    bool allSkippedAlready = job.registerGallery && !job.clearExistingEntries;
    if (job.registerGallery) {
        QVector<FaceDetectGalleryEntry> entries = registrationEntries;
        if (entries.isEmpty() && !job.bundle.extractRoot.isEmpty()) {
            entries = FaceDetectTestData::registrationEntriesForBundle(
                    job.bundle.extractRoot);
        }

        if (job.clearExistingEntries) {
            store.clear();
        }

        const int total = entries.size();
        for (int i = 0; i < total; ++i) {
            const FaceDetectGalleryEntry& entry = entries.at(i);

            if (!job.clearExistingEntries) {
                bool alreadyRegistered = false;
                for (const FaceRegistryEntry& existing : store.entries()) {
                    if (existing.name.compare(entry.name,
                                              Qt::CaseInsensitive) == 0) {
                        alreadyRegistered = true;
                        break;
                    }
                }
                if (alreadyRegistered) {
                    progress.advance(
                            kWeightRegisterEmbed,
                            tr("Skipped %1 — already registered (%2/%3).")
                                    .arg(entry.name)
                                    .arg(i + 1)
                                    .arg(total));
                    continue;
                }
            }

            allSkippedAlready = false;

            progress.report(progress.completed(),
                            tr("Registering %1 (%2/%3) — embedding…")
                                    .arg(entry.name)
                                    .arg(i + 1)
                                    .arg(total));

            QString imagePath = entry.imagePath;
            if (!job.bundle.extractRoot.isEmpty()) {
                const QString fixed =
                        FaceDetectTestData::fixedRegistrationImagePath(
                                job.bundle.extractRoot, entry.name);
                if (!fixed.isEmpty()) {
                    imagePath = fixed;
                }
            }

            QImageReader reader(imagePath);
            reader.setAutoTransform(true);
            const QImage thumb = reader.read();

            std::vector<float> emb;
            QString err;
            bool usedTemplate = false;
            bool ok = false;
            QString registeredPath = imagePath;

            const QStringList candidates =
                    !job.bundle.extractRoot.isEmpty()
                            ? registrationCandidatesForName(
                                      job.bundle.extractRoot, entry.name)
                            : QStringList{imagePath};

            for (const QString& candidate : candidates) {
                if (candidate.isEmpty()) continue;
                bool candidateTemplate = false;
                std::vector<float> candidateEmb;
                if (!FaceDetectEmbed::embedImagePathDetectAligned(
                            ctx, candidate, &candidateEmb, 0.0f,
                            &candidateTemplate)) {
                    continue;
                }
                if (!candidateTemplate) {
                    emb = std::move(candidateEmb);
                    registeredPath = candidate;
                    ok = true;
                    usedTemplate = false;
                    break;
                }
                if (!ok) {
                    emb = std::move(candidateEmb);
                    registeredPath = candidate;
                    ok = true;
                    usedTemplate = true;
                }
            }

            if (!ok) {
                err = QString::fromUtf8(
                        aicore_facedetect_last_error(ctx)
                                ? aicore_facedetect_last_error(ctx)
                                : "embed failed");
            }
            if (ok && registeredPath != imagePath) {
                emit logMessage(
                        tr("[Registry] %1: using registration photo %2")
                                .arg(entry.name,
                                     QFileInfo(registeredPath).fileName()));
            }
            if (ok && usedTemplate) {
                emit logMessage(
                        tr("[Registry] %1: warning — template fallback embed "
                           "(%2); try query portrait for better matches.")
                                .arg(entry.name,
                                     QFileInfo(registeredPath).fileName()));
            }
            if (!ok) {
                emit logMessage(tr("[Registry] %1: %2").arg(entry.name, err));
                progress.advance(kWeightRegisterEmbed,
                                 tr("Register skipped %1 (%2/%3).")
                                         .arg(entry.name)
                                         .arg(i + 1)
                                         .arg(total));
                continue;
            }

            FaceRegistryEntry reg;
            reg.name = entry.name;
            reg.modelFile = QFileInfo(job.modelPath).fileName();
            reg.embedDim = static_cast<int>(emb.size());
            reg.embedding = std::move(emb);
            reg.thumbnail = thumb;
            if (store.addEntry(std::move(reg))) {
                ++registered;
                emit logMessage(
                        tr("[Registry] Registered '%1'.").arg(entry.name));
            }
            progress.advance(kWeightRegisterEmbed, tr("Registered %1 (%2/%3).")
                                                           .arg(entry.name)
                                                           .arg(i + 1)
                                                           .arg(total));
        }
    }

    int authFaces = 0;
    int authMatched = 0;
    if (job.runVerify && !job.bundle.authProbeImage.isEmpty()) {
        const QString probePath = job.bundle.authProbeImage;
        progress.report(progress.completed(),
                        tr("Detecting faces in probe image…"));
        QImage rgb = FaceDetectEmbed::loadRgbForInference(probePath);
        const std::vector<FaceDetectBox> boxes =
                FaceDetectEmbed::detectBoxesFromRgb(ctx, rgb);

        if (boxes.empty()) {
            progress.advance(
                    kWeightVerifyDetect,
                    tr("No faces detected — fallback embed on probe…"));
            std::vector<float> emb;
            QString err;
            if (FaceDetectEmbed::embedImagePathWithFallback(
                        ctx, probePath, &emb, job.minDetectionScore)) {
                authFaces = 1;
                progress.advance(kWeightVerifyEmbed,
                                 tr("Matching probe face…"));
                if (store.bestMatch(emb, job.authThreshold).has_value())
                    ++authMatched;
            } else {
                progress.advance(kWeightVerifyEmbed,
                                 tr("Verify embed failed."));
            }
        } else {
            authFaces = static_cast<int>(boxes.size());
            const int plannedVerifyFaces = kVerifyFaceEstimate;
            if (authFaces > plannedVerifyFaces) {
                progress.bumpTotal((authFaces - plannedVerifyFaces) *
                                   kWeightVerifyEmbed);
            } else if (authFaces < plannedVerifyFaces) {
                progress.adjustTotal(-(plannedVerifyFaces - authFaces) *
                                     kWeightVerifyEmbed);
            }
            progress.advance(
                    kWeightVerifyDetect,
                    tr("Detected %1 face(s) — matching…").arg(authFaces));

            for (int i = 0; i < authFaces; ++i) {
                const FaceDetectBox& box = boxes[static_cast<size_t>(i)];
                progress.report(progress.completed(),
                                tr("Verifying face %1/%2 (det %3) — embedding…")
                                        .arg(i + 1)
                                        .arg(authFaces)
                                        .arg(box.score, 0, 'f', 3));
                if (box.score < job.minDetectionScore) {
                    progress.advance(kWeightVerifyEmbed,
                                     tr("Face %1/%2 skipped (low det score).")
                                             .arg(i + 1)
                                             .arg(authFaces));
                    continue;
                }
                std::vector<float> emb;
                if (rgb.isNull() ||
                    !FaceDetectEmbed::embedFaceBoxFromFrame(
                            ctx, rgb, box, job.minDetectionScore, &emb)) {
                    progress.advance(kWeightVerifyEmbed,
                                     tr("Face %1/%2 embed failed.")
                                             .arg(i + 1)
                                             .arg(authFaces));
                    continue;
                }
                const bool matched =
                        store.bestMatch(emb, job.authThreshold).has_value();
                if (matched) ++authMatched;
                progress.advance(
                        kWeightVerifyEmbed,
                        tr("Face %1/%2 %3.")
                                .arg(i + 1)
                                .arg(authFaces)
                                .arg(matched ? tr("matched") : tr("no match")));
            }
        }
        emit logMessage(
                tr("[Test data] Verify: %1 face(s), %2 matched (threshold %3).")
                        .arg(authFaces)
                        .arg(authMatched)
                        .arg(job.authThreshold, 0, 'f', 2));
    }

    progress.report(progress.total(), tr("Test data setup complete."));
    const bool registerOk = !job.registerGallery || registered > 0 ||
                            (allSkippedAlready && registerCount > 0);
    emit finished(registerOk, registered, authFaces, authMatched);
#endif
}
