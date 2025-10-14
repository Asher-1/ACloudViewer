// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UI_TRIANGLE_PAINTER_H_
#define COLMAP_SRC_UI_TRIANGLE_PAINTER_H_

#include <QtCore>
#include <QtOpenGL>

#include "ui/point_painter.h"

namespace colmap {

class TrianglePainter {
public:
    TrianglePainter();
    ~TrianglePainter();

    struct Data {
        Data() {}
        Data(const PointPainter::Data& p1,
             const PointPainter::Data& p2,
             const PointPainter::Data& p3)
            : point1(p1), point2(p2), point3(p3) {}

        PointPainter::Data point1;
        PointPainter::Data point2;
        PointPainter::Data point3;
    };

    void Setup();
    void Upload(const std::vector<TrianglePainter::Data>& data);
    void Render(const QMatrix4x4& pmv_matrix);

private:
    QOpenGLShaderProgram shader_program_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_;

    size_t num_geoms_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_TRIANGLE_PAINTER_H_
