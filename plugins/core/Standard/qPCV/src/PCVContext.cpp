// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Based on CloudCompare's PCVContext (EDF R&D / TELECOM ParisTech)
// Inspired from ShadeVis' "GLAccumPixel" (Cignoni et al.)

#include "PCVContext.h"

// cloudViewer
#include <GenericTriangle.h>
#include <ecvGLMatrix.h>

// Qt OpenGL
#include <QCoreApplication>
#include <QOffscreenSurface>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QSurfaceFormat>
#include <QWindow>

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QOpenGLVersionFunctionsFactory>
#endif
#include <QOpenGLFunctions_2_1>

// system
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

using namespace cloudViewer;

static constexpr double ZTWIST = 1.0e-3;

static inline void GLVertex3v(const float* v, QOpenGLFunctions_2_1* f) {
    f->glVertex3fv(v);
}
static inline void GLVertex3v(const double* v, QOpenGLFunctions_2_1* f) {
    f->glVertex3dv(v);
}

static inline QOpenGLFunctions_2_1* getGLFunctions(QOpenGLContext* ctx) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_2_1>(ctx);
#else
    return ctx->versionFunctions<QOpenGLFunctions_2_1>();
#endif
}

// LookAt matrix (same as gluLookAt / ccGL::LookAt in CloudCompare)
static ccGLMatrixd PCVLookAt(const CCVector3d& eye,
                             const CCVector3d& center,
                             const CCVector3d& up) {
    CCVector3d forward = eye - center;
    forward.normalize();

    CCVector3d left = up.cross(forward);
    left.normalize();

    CCVector3d upFixed = forward.cross(left);

    ccGLMatrixd mr;
    mr.data()[0] = left.x;
    mr.data()[4] = left.y;
    mr.data()[8] = left.z;
    mr.data()[1] = upFixed.x;
    mr.data()[5] = upFixed.y;
    mr.data()[9] = upFixed.z;
    mr.data()[2] = forward.x;
    mr.data()[6] = forward.y;
    mr.data()[10] = forward.z;
    mr.data()[3] = 0;
    mr.data()[7] = 0;
    mr.data()[11] = 0;
    mr.data()[12] = 0;
    mr.data()[13] = 0;
    mr.data()[14] = 0;
    mr.data()[15] = 1;

    ccGLMatrixd mt;
    mt.toIdentity();
    mt.data()[12] = -eye.x;
    mt.data()[13] = -eye.y;
    mt.data()[14] = -eye.z;

    return mr * mt;
}

// Project 3D point to window coordinates (same as gluProject / ccGL::Project)
static bool PCVProject(const CCVector3d& input3D,
                       const double* modelview,
                       const double* projection,
                       const int* viewport,
                       CCVector3d& output2D) {
    double Pm[4];
    Pm[0] = modelview[0] * input3D.x + modelview[4] * input3D.y +
            modelview[8] * input3D.z + modelview[12];
    Pm[1] = modelview[1] * input3D.x + modelview[5] * input3D.y +
            modelview[9] * input3D.z + modelview[13];
    Pm[2] = modelview[2] * input3D.x + modelview[6] * input3D.y +
            modelview[10] * input3D.z + modelview[14];
    Pm[3] = modelview[3] * input3D.x + modelview[7] * input3D.y +
            modelview[11] * input3D.z + modelview[15];

    double Pp[4];
    Pp[0] = projection[0] * Pm[0] + projection[4] * Pm[1] +
            projection[8] * Pm[2] + projection[12] * Pm[3];
    Pp[1] = projection[1] * Pm[0] + projection[5] * Pm[1] +
            projection[9] * Pm[2] + projection[13] * Pm[3];
    Pp[2] = projection[2] * Pm[0] + projection[6] * Pm[1] +
            projection[10] * Pm[2] + projection[14] * Pm[3];
    Pp[3] = projection[3] * Pm[0] + projection[7] * Pm[1] +
            projection[11] * Pm[2] + projection[15] * Pm[3];

    if (Pp[3] == 0.0) {
        return false;
    }

    Pp[0] /= Pp[3];
    Pp[1] /= Pp[3];
    Pp[2] /= Pp[3];

    output2D.x = (1.0 + Pp[0]) / 2 * viewport[2];
    output2D.y = (1.0 + Pp[1]) / 2 * viewport[3];
    output2D.z = (1.0 + Pp[2]) / 2;

    return true;
}

PCVContext::PCVContext()
    : m_vertices(nullptr),
      m_mesh(nullptr),
      m_entityDiagonal(0),
      m_glSurface(nullptr),
      m_glContext(nullptr),
      m_pixBuffer(nullptr),
      m_fbo(nullptr),
      m_width(0),
      m_height(0),
      m_meshIsClosed(false) {}

PCVContext::~PCVContext() {
    if (m_glContext && m_glSurface) {
        m_glContext->makeCurrent(m_glSurface);
    }
    delete m_fbo;
    delete m_pixBuffer;
    if (m_glContext) {
        m_glContext->doneCurrent();
        delete m_glContext;
        m_glContext = nullptr;
    }
    delete m_glSurface;
}

bool PCVContext::init(unsigned W,
                      unsigned H,
                      cloudViewer::GenericCloud* cloud,
                      cloudViewer::GenericMesh* mesh,
                      bool closedMesh) {
    assert(!m_pixBuffer);

    m_meshIsClosed = (closedMesh || !mesh);

    unsigned size = W * H;
    try {
        m_snapZ.resize(size, 0);
        if (!m_meshIsClosed) {
            m_snapC.resize(size * 4, 0);
        }
    } catch (const std::bad_alloc&) {
        return false;
    }

    m_width = W;
    m_height = H;

    associateToEntity(cloud, mesh);

    if (!glInit()) {
        return false;
    }

    if (!makeCurrent()) {
        return false;
    }

    auto* functions = getGLFunctions(m_glContext);
    if (!functions) {
        return false;
    }
    functions->initializeOpenGLFunctions();
    functions->glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    functions->glPixelStorei(GL_PACK_ALIGNMENT, 1);
    functions->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    QOpenGLFramebufferObjectFormat fboFmt;
    fboFmt.setAttachment(QOpenGLFramebufferObject::Depth);
    fboFmt.setInternalTextureFormat(GL_RGBA8);
    m_fbo = new QOpenGLFramebufferObject(static_cast<int>(W),
                                         static_cast<int>(H), fboFmt);
    if (!m_fbo->isValid()) {
        delete m_fbo;
        m_fbo = nullptr;
        return false;
    }

    m_pixBuffer = new QOpenGLBuffer(QOpenGLBuffer::PixelPackBuffer);
    if (!m_pixBuffer->create()) {
        delete m_pixBuffer;
        m_pixBuffer = nullptr;
        return false;
    }

    m_pixBuffer->setUsagePattern(QOpenGLBuffer::StreamRead);

    if (!m_pixBuffer->bind()) {
        return false;
    }

    m_pixBuffer->allocate(static_cast<int>(static_cast<size_t>(W) * H * sizeof(float)));
    m_pixBuffer->release();

    return true;
}

void PCVContext::associateToEntity(GenericCloud* cloud, GenericMesh* mesh) {
    m_vertices = nullptr;
    m_entityDiagonal = 0;
    m_entityCenter = CCVector3(0, 0, 0);

    if (!cloud) {
        assert(false);
        return;
    }

    m_vertices = cloud;
    m_mesh = mesh;

    CCVector3 bbMin, bbMax;
    m_vertices->getBoundingBox(bbMin, bbMax);
    m_entityDiagonal = (bbMax - bbMin).norm();
    m_entityCenter = (bbMax + bbMin) / 2;
}

bool PCVContext::makeCurrent() {
    if (!m_glContext || !m_glSurface) {
        return false;
    }
    return m_glContext->makeCurrent(m_glSurface);
}

bool PCVContext::glInit() {
    if (!m_glContext) {
        m_glContext = new QOpenGLContext;

        QSurfaceFormat fmt;
        fmt.setDepthBufferSize(24);
        fmt.setStencilBufferSize(8);
        fmt.setVersion(2, 1);
        fmt.setProfile(QSurfaceFormat::CompatibilityProfile);
        m_glContext->setFormat(fmt);

        QOpenGLContext* shareCtx = QOpenGLContext::globalShareContext();
        if (shareCtx) {
            m_glContext->setShareContext(shareCtx);
        }

        if (!m_glContext->create()) {
            delete m_glContext;
            m_glContext = nullptr;
            return false;
        }
    }

    if (!m_glSurface) {
        QOffscreenSurface* surface = new QOffscreenSurface;
        surface->setFormat(m_glContext->format());
        surface->create();
        m_glSurface = surface;
    }

    return true;
}

void PCVContext::drawEntity() {
    auto* functions = getGLFunctions(m_glContext);
    if (!functions) {
        return;
    }

    if (m_mesh) {
        unsigned nTri = m_mesh->size();
        m_mesh->placeIteratorAtBeginning();

        functions->glBegin(GL_TRIANGLES);
        for (unsigned i = 0; i < nTri; ++i) {
            const GenericTriangle* t = m_mesh->_getNextTriangle();
            GLVertex3v(t->_getA()->u, functions);
            GLVertex3v(t->_getB()->u, functions);
            GLVertex3v(t->_getC()->u, functions);
        }
        functions->glEnd();
    } else if (m_vertices) {
        unsigned nPts = m_vertices->size();
        m_vertices->placeIteratorAtBeginning();

        functions->glBegin(GL_POINTS);
        for (unsigned i = 0; i < nPts; ++i) {
            GLVertex3v(m_vertices->getNextPoint()->u, functions);
        }
        functions->glEnd();
    }
}

//! Renders the entity from \a viewDir into the offscreen buffer and increments per-vertex visibility using depth/occlusion tests.
int PCVContext::glAccumPixel(std::vector<int>& visibilityCount,
                             const CCVector3d& viewDir) {
    if (!m_pixBuffer || !m_pixBuffer->isCreated()) return -1;
    if (!m_fbo) return -1;
    if (!m_vertices) return -1;
    if (m_vertices->size() != visibilityCount.size()) return -1;

    assert(m_snapZ.data());

    if (!makeCurrent()) {
        return -2;
    }

    auto* functions = getGLFunctions(m_glContext);
    if (!functions) {
        return -3;
    }

    m_fbo->bind();

    functions->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    functions->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    functions->glDepthRange(2.0 * ZTWIST, 1.0);

    functions->glPointSize(1.0f);
    functions->glDisable(GL_BLEND);
    functions->glEnable(GL_DEPTH_TEST);
    functions->glEnable(GL_CULL_FACE);
    functions->glDisable(GL_LIGHTING);

    double projectionMat[OPENGL_MATRIX_SIZE];
    {
        functions->glMatrixMode(GL_PROJECTION);
        functions->glLoadIdentity();
        double ar = static_cast<double>(m_height) / m_width;
        double xMax = m_entityDiagonal / 2;
        double yMax = xMax * ar;
        functions->glOrtho(-xMax, xMax, -yMax, yMax, -xMax, xMax);
        functions->glGetDoublev(GL_PROJECTION_MATRIX, projectionMat);
    }

    ccGLMatrixd viewMat;
    viewMat.toIdentity();
    {
        CCVector3d U(0.0, 0.0, 1.0);
        if (1.0 - std::abs(viewDir.dot(U)) < 1.0e-4) {
            U.y = 1.0;
            U.z = 0.0;
        }

        viewMat = PCVLookAt(m_entityCenter.toDouble() - viewDir,
                            m_entityCenter.toDouble(), U);

        functions->glMatrixMode(GL_MODELVIEW);
        functions->glLoadIdentity();
        functions->glMultMatrixd(viewMat.data());
    }

    functions->glColor3ub(255, 255, 0);

    GLboolean drawColor = m_meshIsClosed ? GL_FALSE : GL_TRUE;
    functions->glColorMask(drawColor, drawColor, drawColor, drawColor);

    int viewPort[4]{0, 0, static_cast<int>(m_width),
                    static_cast<int>(m_height)};
    functions->glViewport(viewPort[0], viewPort[1], viewPort[2], viewPort[3]);

    functions->glCullFace(GL_BACK);
    drawEntity();

    if (m_mesh && !m_meshIsClosed) {
        functions->glCullFace(GL_FRONT);
        drawEntity();

        if (!m_pixBuffer->bind()) {
            return -4;
        }
        functions->glReadPixels(viewPort[0], viewPort[1], viewPort[2],
                                viewPort[3], GL_RGBA, GL_UNSIGNED_BYTE,
                                nullptr);
        m_pixBuffer->read(0, m_snapC.data(),
                          static_cast<int>(m_width * m_height * 4));
        m_pixBuffer->release();
    }

    {
        if (!m_pixBuffer->bind()) {
            return -4;
        }
        functions->glReadPixels(viewPort[0], viewPort[1], viewPort[2],
                                viewPort[3], GL_DEPTH_COMPONENT, GL_FLOAT,
                                nullptr);
        m_pixBuffer->read(0, m_snapZ.data(),
                          static_cast<int>(static_cast<size_t>(m_width) * m_height * sizeof(float)));
        m_pixBuffer->release();
    }

    int count = 0;
    int sx4 = (m_width << 2);

    unsigned nVert = m_vertices->size();
    m_vertices->placeIteratorAtBeginning();
    for (unsigned i = 0; i < nVert; ++i) {
        const CCVector3* P = m_vertices->getNextPoint();

        CCVector3d P2D;
        PCVProject(P->toDouble(), viewMat.data(), projectionMat, viewPort, P2D);

        int txi = static_cast<int>(floor(P2D.x));
        int tyi = static_cast<int>(floor(P2D.y));
        if (txi >= 0 && txi < static_cast<int>(m_width) && tyi >= 0 &&
            tyi < static_cast<int>(m_height)) {
            int dec = txi + tyi * static_cast<int>(m_width);
            uint8_t col = 1;

            if (!m_meshIsClosed) {
                const uint8_t* pix =
                        m_snapC.data() + (static_cast<size_t>(dec) << 2);
                uint8_t c1 = std::max(pix[0], pix[4]);
                pix += sx4;
                uint8_t c2 = std::max(pix[0], pix[4]);
                col = std::max(c1, c2);
            }

            if (col != 0) {
                if (P2D.z < static_cast<double>(m_snapZ[dec])) {
                    assert(i < visibilityCount.size());
                    ++visibilityCount[i];
                    ++count;
                }
            }
        }
    }

    m_fbo->release();

    assert(m_glContext);
    m_glContext->doneCurrent();

    return count;
}
