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

signals:
    void logMessage(const QString& message);
    void stageChanged(int stage, int step, int total);
    void resultReady(const Sam3dRunResult& result);
    //! Multi-object scene envelope (mask-directory runs only).
    void sceneReady(const Sam3dSceneResult& result);
    void taskFinished(bool success);

private:
    bool decodeInputImage();
    bool resolveSam3dModels();
    bool applyRmbgMask();
    bool runGeneration();
    //! Scene mode: one independent single-object run per mask file, then the
    //! AICore scene assembler composes them with the official make_scene
    //! pose semantics. Each textured object is baked and written to its own
    //! GLB file (bounded memory: GLB bytes are released after the write).
    bool runSceneGeneration();
    //! Copies the typed generate() result into the plugin envelope.
    void fillRunResult(aicore_sam3d_ctx* ctx,
                       aicore_sam3d_result* sam3d,
                       Sam3dRunResult& result);
    //! Nearest-splat vertex colors -> per-vertex PBR -> shared AICore bake
    //! pipeline (xatlas UV atlas) -> GLB bytes. Returns false on bake failure
    //! (logged; the generation result stays valid).
    bool bakeTexturedGlb(Sam3dRunResult& result);
    //! Scene variant: bakes with the vertices transformed into the scene
    //! frame (official make_scene position action) and releases the GLB
    //! bytes after the file write (the path is the DB import source).
    bool bakeSceneObjectGlb(Sam3dRunResult& result, int objectIndex);

    Sam3dDialog::Settings m_settings;
    std::atomic<bool> m_cancel{false};
    // Non-premultiplied ARGB32 pipeline input; the object mask, when RMBG
    // runs, is merged straight into its alpha bytes. The image-view borrow
    // maps this storage directly (little-endian BGRA8 + bytesPerLine).
    QImage m_sourceImage;
    QByteArray m_rmbgModel;  // keep-alive for AICore string copies
    QString m_modelsDir;
    QString m_error;
};
