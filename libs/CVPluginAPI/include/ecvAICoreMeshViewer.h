// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QString>
#include <QVector>

#include "CVPluginAPI.h"

/** Modal orbit viewer for one AICore pipeline-step mesh, shared by the mesh
 *  plugins (qTrellis, qInstantMesh). Left-drag rotates, the wheel zooms,
 *  double-click resets the view. The buffers are shared (QVector
 *  copy-on-write), never deep-copied. */
class CVPLUGIN_LIB_API ecvAICoreMeshViewerDialog : public QDialog {
    Q_OBJECT

public:
    /** \p pbr carries per-vertex base colour (6 floats: rgb + metallic,
     *  roughness, alpha — only rgb is shown) when textured; empty renders
     *  the amber geometry shading used by the strip thumbnails. \p normals
     *  carries per-vertex normals (3 floats); missing or zero normals shade
     *  flat. */
    ecvAICoreMeshViewerDialog(const QVector<float>& verts,
                              const QVector<float>& normals,
                              const QVector<int>& tris,
                              const QVector<float>& pbr,
                              bool textured,
                              const QString& title,
                              QWidget* parent = nullptr);
};
