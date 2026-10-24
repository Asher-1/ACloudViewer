// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TrellisMeshViewer.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QSurfaceFormat>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <cmath>
#include <vector>

namespace {

// Qt 5 uses QPointF localPos(); Qt 6 renamed it to position().
QPointF eventPos(const QMouseEvent* e) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return e->position();
#else
    return e->localPos();
#endif
}

// Orbit GL viewport: one interleaved VBO (pos + normal + colour), one IBO, a
// phong-lite shader. A single static draw call handles the multi-million-
// triangle TRELLIS meshes comfortably.
class MeshGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
public:
    MeshGLWidget(const QVector<float>& verts,
                 const QVector<float>& normals,
                 const QVector<int>& tris,
                 const QVector<float>& pbr,
                 bool textured,
                 QWidget* parent)
        : QOpenGLWidget(parent),
          m_verts(verts),
          m_normals(normals),
          m_tris(tris),
          m_pbr(pbr),
          m_textured(textured) {
        // Core-profile 3.3 keeps the shader stack predictable across
        // drivers; QOpenGLWidget picks up the format before initializeGL.
        QSurfaceFormat fmt = format();
        fmt.setVersion(3, 3);
        fmt.setProfile(QSurfaceFormat::CoreProfile);
        fmt.setDepthBufferSize(24);
        setFormat(fmt);
    }

    void resetView() {
        m_yaw = 35.0f;
        m_pitch = 20.0f;
        m_dist = m_fitDist;
        update();
    }

protected:
    void initializeGL() override {
        initializeOpenGLFunctions();
        glClearColor(0.10f, 0.12f, 0.15f, 1.0f);
        glEnable(GL_DEPTH_TEST);

        const int nv = m_verts.size() / 3;
        const int nt = m_tris.size() / 3;
        const bool hasN = m_normals.size() >= nv * 3;
        const bool hasC = m_textured && m_pbr.size() >= nv * 6;

        // Interleaved pos(3) + normal(3) + colour(3); centre the mesh on its
        // bounding box so the orbit pivots through the object.
        std::vector<float> data((size_t)nv * 9);
        float cmin[3] = {1e9f, 1e9f, 1e9f}, cmax[3] = {-1e9f, -1e9f, -1e9f};
        for (int i = 0; i < nv; ++i) {
            float* d = data.data() + (size_t)i * 9;
            for (int k = 0; k < 3; ++k) {
                const float p = m_verts[3 * i + k];
                d[k] = p;
                cmin[k] = std::min(cmin[k], p);
                cmax[k] = std::max(cmax[k], p);
            }
            for (int k = 0; k < 3; ++k)
                d[3 + k] = hasN ? m_normals[3 * i + k] : 0.0f;
            float r = 0.910f, g = 0.651f, b = 0.384f;  // strip amber fallback
            if (hasC) {
                r = m_pbr[6 * i + 0];
                g = m_pbr[6 * i + 1];
                b = m_pbr[6 * i + 2];
            }
            d[6] = r;
            d[7] = g;
            d[8] = b;
        }
        for (int k = 0; k < 3; ++k) m_center[k] = 0.5f * (cmin[k] + cmax[k]);
        float radius = 0.0f;
        for (int k = 0; k < 3; ++k)
            radius = std::max(radius, cmax[k] - cmin[k]);
        m_fitDist = radius * 1.9f + 0.35f;
        m_dist = m_fitDist;

        m_vao.create();
        m_vao.bind();
        m_vbo.create();
        m_vbo.bind();
        m_vbo.allocate(data.data(),
                       static_cast<int>(data.size() * sizeof(float)));
        m_ibo.create();
        m_ibo.bind();
        m_ibo.allocate(m_tris.constData(),
                       static_cast<int>(m_tris.size() * sizeof(int)));

        m_program.create();
        m_program.addShaderFromSourceCode(
                QOpenGLShader::Vertex,
                "#version 330 core\n"
                "layout(location = 0) in vec3 aPos;\n"
                "layout(location = 1) in vec3 aNormal;\n"
                "layout(location = 2) in vec3 aColor;\n"
                "uniform mat4 uMvp;\n"
                "out vec3 vN;\n"
                "out vec3 vC;\n"
                "void main() {\n"
                "    vN = aNormal;\n"
                "    vC = aColor;\n"
                "    gl_Position = uMvp * vec4(aPos, 1.0);\n"
                "}");
        m_program.addShaderFromSourceCode(
                QOpenGLShader::Fragment,
                "#version 330 core\n"
                "in vec3 vN;\n"
                "in vec3 vC;\n"
                "out vec4 frag;\n"
                "void main() {\n"
                "    vec3 L = normalize(vec3(0.35, 0.6, 0.72));\n"
                "    float d = abs(dot(normalize(vN + 1e-6), L));\n"
                "    frag = vec4(vC * (0.25 + 0.75 * d), 1.0);\n"
                "}");
        m_program.link();
        m_program.bind();
        m_uMvp = m_program.uniformLocation("uMvp");
        m_program.enableAttributeArray(0);
        m_program.enableAttributeArray(1);
        m_program.enableAttributeArray(2);
        m_program.setAttributeBuffer(0, GL_FLOAT, 0, 3, 9 * sizeof(float));
        m_program.setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3,
                                     9 * sizeof(float));
        m_program.setAttributeBuffer(2, GL_FLOAT, 6 * sizeof(float), 3,
                                     9 * sizeof(float));
        m_vao.release();
        m_program.release();
        m_vbo.release();
        m_ibo.release();
    }

    void resizeGL(int w, int h) override {
        m_aspect = float(h > 0 ? w : 1) / float(h > 0 ? h : 1);
    }

    void paintGL() override {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (m_tris.isEmpty()) return;
        QMatrix4x4 proj;
        proj.perspective(45.0, double(m_aspect), 0.01, 100.0);
        const float cp = std::cos(m_pitch * float(M_PI) / 180.0f);
        const float sp = std::sin(m_pitch * float(M_PI) / 180.0f);
        const float cy = std::cos(m_yaw * float(M_PI) / 180.0f);
        const float sy = std::sin(m_yaw * float(M_PI) / 180.0f);
        const float eye[3] = {m_center[0] + m_dist * cp * sy,
                              m_center[1] + m_dist * sp,
                              m_center[2] + m_dist * cp * cy};
        QMatrix4x4 view;
        view.lookAt(QVector3D(eye[0], eye[1], eye[2]),
                    QVector3D(m_center[0], m_center[1], m_center[2]),
                    QVector3D(0, 1, 0));
        QMatrix4x4 mvp = proj * view;

        m_program.bind();
        m_program.setUniformValue(m_uMvp, mvp);
        m_vao.bind();
        glDrawElements(GL_TRIANGLES, m_tris.size(), GL_UNSIGNED_INT, nullptr);
        m_vao.release();
        m_program.release();
    }

    void mousePressEvent(QMouseEvent* e) override {
        m_lastPos = eventPos(e);
        if (e->button() == Qt::LeftButton) m_orbiting = true;
    }
    void mouseReleaseEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton) m_orbiting = false;
        m_lastPos = eventPos(e);
    }
    void mouseMoveEvent(QMouseEvent* e) override {
        if (!m_orbiting) return;
        const QPointF d = eventPos(e) - m_lastPos;
        m_lastPos = eventPos(e);
        m_yaw += float(d.x()) * 0.4f;
        m_pitch = std::clamp(m_pitch + float(d.y()) * 0.4f, -85.0f, 85.0f);
        update();
    }
    void wheelEvent(QWheelEvent* e) override {
        m_dist *= std::pow(0.92f, float(e->angleDelta().y()) / 120.0f);
        m_dist = std::clamp(m_dist, m_fitDist * 0.15f, m_fitDist * 8.0f);
        update();
    }
    void mouseDoubleClickEvent(QMouseEvent* e) override {
        resetView();
        QOpenGLWidget::mouseDoubleClickEvent(e);
    }

private:
    QVector<float> m_verts;
    QVector<float> m_normals;
    QVector<int> m_tris;
    QVector<float> m_pbr;
    bool m_textured;
    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer m_ibo{QOpenGLBuffer::IndexBuffer};
    QOpenGLShaderProgram m_program;
    int m_uMvp = -1;
    float m_center[3] = {0, 0, 0};
    float m_fitDist = 2.0f;
    float m_dist = 2.0f;
    float m_yaw = 35.0f, m_pitch = 20.0f;
    float m_aspect = 1.0f;
    bool m_orbiting = false;
    QPointF m_lastPos;
};

}  // namespace

TrellisMeshViewerDialog::TrellisMeshViewerDialog(const QVector<float>& verts,
                                                 const QVector<float>& normals,
                                                 const QVector<int>& tris,
                                                 const QVector<float>& pbr,
                                                 bool textured,
                                                 const QString& title,
                                                 QWidget* parent)
    : QDialog(parent) {
    setWindowTitle(title);
    resize(680, 580);
    auto* lay = new QVBoxLayout(this);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->setSpacing(0);
    auto* hint = new QLabel(
            tr("  Left-drag: rotate | wheel: zoom | double-click: reset"),
            this);
    hint->setStyleSheet(
            "color: #98A4B0; background: #22262B; padding: 3px; font-size: "
            "11px;");
    lay->addWidget(hint);
    lay->addWidget(new MeshGLWidget(verts, normals, tris, pbr, textured, this),
                   1);
}
