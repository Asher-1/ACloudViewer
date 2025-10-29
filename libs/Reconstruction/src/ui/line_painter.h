// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtOpenGL>

#include "ui/point_painter.h"

namespace colmap {

class LinePainter {
public:
    LinePainter();
    ~LinePainter();

    struct Data {
        Data() {}
        Data(const PointPainter::Data& p1, const PointPainter::Data& p2)
            : point1(p1), point2(p2) {}

        PointPainter::Data point1;
        PointPainter::Data point2;
    };

    void Setup();
    void Upload(const std::vector<LinePainter::Data>& data);
    void Render(const QMatrix4x4& pmv_matrix,
                const int width,
                const int height,
                const float line_width);

private:
    QOpenGLShaderProgram shader_program_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_;

    size_t num_geoms_;
};

}  // namespace colmap
