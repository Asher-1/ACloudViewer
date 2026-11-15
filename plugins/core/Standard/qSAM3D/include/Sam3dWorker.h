// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QThread>
#include <atomic>

#include "Sam3dDialog.h"
#include "aicore/sam3d_capi.h"

class Sam3dWorker : public QThread {
    Q_OBJECT

public:
    explicit Sam3dWorker(QObject* parent = nullptr);
    ~Sam3dWorker() override;

    void configure(const Sam3dDialog::Settings& settings);
    QString error() const { return m_error; }
    void run() override;

    void requestCancel();

    //! Progress model: per-stage spans of one object's 0..100 range, sliced
    //! by the object base/span in scene mode. Public: the C-callback
    //! trampoline (a free function) maps stage events through it.
    int progressPercentFor(int stage, int step, int total) const;

signals:
    void logMessage(const QString& message);
    void stageChanged(int stage, int step, int total);
    //! Coarse-grained progress in percent (0..100). Scene mode slices the
    //! span per object; flow stages interpolate by step/total.
    void progressChanged(int percent);
    void resultReady(const Sam3dRunResult& result);
    //! Incremental scene delivery: emitted as soon as ONE object finished
    //! its full pipeline (generate -> scene placement -> bake), so the GUI
    //! can import it into the DB tree immediately instead of waiting for
    //! the whole scene. Carries the object already placed in the scene
    //! frame with the GLB written and splatRgb released.
    void objectReady(const Sam3dRunResult& object);
    //! Total resolved object count, emitted once after resolveSceneMasks
    //! (drives the GUI-side "object i/N" progress annotations).
    void sceneObjectTotalChanged(int total);
    //! Scene-mode diagnostic (opt-out): every accepted object mask composited
    //! over the source image, one color + label per object. Emitted once
    //! after resolveSceneMasks, BEFORE the per-object runs start — it tells
    //! the user exactly which objects the segmentation accepted while the
    //! reconstruction is still running.
    void segmentationPreviewReady(const QImage& preview, int maskCount);
    //! Multi-object scene envelope (mask-directory runs only).
    void sceneReady(const Sam3dSceneResult& result);
    void taskFinished(bool success);

private:
    //! Scene mode is requested: concept prompts (automatic sam3
    //! segmentation) and/or the explicit mask directory.
    bool sceneModeRequested() const {
        return !m_settings.masksDir.isEmpty() ||
               !m_settings.conceptPrompts.isEmpty();
    }
    bool decodeInputImage();
    bool resolveSam3dModels();
    bool applyRmbgMask();
    bool runGeneration();
    //! Scene mask source: concept prompts drive the AICore sam3 task's
    //! text-prompted multi-instance segmentation (fully in-memory); an
    //! explicit mask directory is the interoperability fallback. Masks come
    //! back deduplicated (greedy IoU) and score-ordered.
    bool resolveSceneMasks(QVector<QImage>& masks);
    //! Two-step workflow, step 1 → step 2: the accepted masks of the last
    //! segmentation are persisted as one RLE stream file plus a fingerprint
    //! (image + prompts + filters + sam3 model). Generate reloads them
    //! verbatim on an exact match and skips the whole SAM3 chain — the
    //! model is not even loaded. Any input change invalidates the cache.
    bool loadPreviewMaskCache(QVector<QImage>& masks);
    void savePreviewMaskCache(const QVector<QImage>& masks);
    //! Human-readable mask source descriptions parallel to the masks vector
    //! (prompt provenance or file names), for logs and metadata.
    QStringList m_sceneMaskLabels;
    //! Head of the global 0..100 progress axis already consumed by the
    //! segmentation; runSceneGeneration slices per-object progress out of
    //! [m_segEndPercent, 100]. 0 when no segmentation ran (mask directory).
    int m_segEndPercent = 0;
    //! Scene mode: one independent single-object run per mask, then the
    //! AICore scene assembler composes them with the official make_scene
    //! pose semantics. Each textured object is baked and written to its own
    //! GLB file (bounded memory: GLB bytes are released after the write).
    bool runSceneGeneration();
    //! Composites the accepted masks over the source image and emits
    //! segmentationPreviewReady (opt-out through Settings::segPreview).
    void emitSegmentationPreview(const QVector<QImage>& masks);
    //! Copies the typed generate() result into the plugin envelope.
    void fillRunResult(aicore_sam3d_ctx* ctx,
                       aicore_sam3d_result* sam3d,
                       Sam3dRunResult& result);
    //! Nearest-splat vertex colors -> per-vertex PBR -> shared AICore bake
    //! pipeline (xatlas UV atlas) -> GLB bytes. Returns false on bake failure
    //! (logged; the generation result stays valid).
    bool bakeTexturedGlb(Sam3dRunResult& result);
    //! Writes result.glb into the user-chosen output directory (opt-in;
    //! empty outputDir never writes). Uses <image>[_<label>].glb naming.
    bool writeGlbFile(Sam3dRunResult& result);
    //! Scene variant: bakes with the vertices transformed into the scene
    //! frame (official make_scene position action) and releases the GLB
    //! bytes after the file write (the path is the DB import source).
    bool bakeSceneObjectGlb(Sam3dRunResult& result, int objectIndex);

    Sam3dDialog::Settings m_settings;
    std::atomic<bool> m_cancel{false};
    // Progress state: the [base, base+span) slice of the global 0..100 range
    // that the CURRENT object occupies (scene mode slices it per object).
    std::atomic<int> m_objBase{0};
    std::atomic<int> m_objSpan{100};
    // Non-premultiplied ARGB32 pipeline input; the object mask, when RMBG
    // runs, is merged straight into its alpha bytes. The image-view borrow
    // maps this storage directly (little-endian BGRA8 + bytesPerLine).
    QImage m_sourceImage;
    QByteArray m_rmbgModel;  // keep-alive for AICore string copies
    QString m_modelsDir;
    QString m_error;
};
