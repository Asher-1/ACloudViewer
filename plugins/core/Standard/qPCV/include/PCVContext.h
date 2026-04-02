// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// cloudViewer
#include <GenericCloud.h>
#include <GenericMesh.h>

// system
#include <cstdint>
#include <vector>

class QSurface;
class QOpenGLBuffer;
class QOpenGLContext;
class QOpenGLFramebufferObject;

//! PCV (Portion de Ciel Visible / Ambient occlusion) OpenGL context
/** Similar to Cignoni's ShadeVis.
    Uses Qt OpenGL off-screen rendering for depth-based visibility testing.
**/
class PCVContext {
public:
    PCVContext();
    virtual ~PCVContext();

    bool init(unsigned W,
              unsigned H,
              cloudViewer::GenericCloud* cloud,
              cloudViewer::GenericMesh* mesh = nullptr,
              bool closedMesh = true);

    int glAccumPixel(std::vector<int>& visibilityCount,
                     const CCVector3d& viewDir);

    bool makeCurrent();

protected:
    bool glInit();
    void drawEntity();
    void associateToEntity(cloudViewer::GenericCloud* cloud,
                           cloudViewer::GenericMesh* mesh = nullptr);

    cloudViewer::GenericCloud* m_vertices;
    cloudViewer::GenericMesh* m_mesh;

    PointCoordinateType m_entityDiagonal;
    CCVector3 m_entityCenter;

    QSurface* m_glSurface;
    QOpenGLContext* m_glContext;
    QOpenGLBuffer* m_pixBuffer;
    QOpenGLFramebufferObject* m_fbo;

    unsigned m_width;
    unsigned m_height;

    std::vector<float> m_snapZ;
    std::vector<uint8_t> m_snapC;

    bool m_meshIsClosed;
};
