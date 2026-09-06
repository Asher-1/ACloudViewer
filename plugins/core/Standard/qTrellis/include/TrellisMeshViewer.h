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

/** Modal orbit viewer for one pipeline-step mesh. Left-drag rotates, the
 *  wheel zooms, double-click resets the view. The buffers are shared
 *  (QVector copy-on-write), never deep-copied. */
class TrellisMeshViewerDialog : public QDialog {
    Q_OBJECT

public:
    /** \p pbr carries per-vertex base colour (6 floats: rgb + metallic,
     *  roughness, alpha — only rgb is shown) when textured; empty renders
     *  the amber geometry shading used by the strip thumbnails. */
    TrellisMeshViewerDialog(const QVector<float>& verts,
                            const QVector<float>& normals,
                            const QVector<int>& tris,
                            const QVector<float>& pbr,
                            bool textured,
                            const QString& title,
                            QWidget* parent = nullptr);
};
