// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "eCV_db.h"
#include "ecvGLMatrix.h"

// CV_CORE_LIB
#include <CVConst.h>
#include <CVToolbox.h>

// SYSTEM
#include <algorithm>
#include <cmath>
#include <iostream>

//! Generic display tools
class ECV_DB_LIB_API ecvGenericDisplayTools : public cloudViewer::CVToolbox {
public:
    //! Default destructor
    virtual ~ecvGenericDisplayTools() {}

    static ecvGenericDisplayTools* GetInstance();
    static void SetInstance(ecvGenericDisplayTools* tool);

public:
    static int FontSizeModifier(int fontSize, float zoomFactor) {
        int scaledFontSize =
                static_cast<int>(std::floor(fontSize * zoomFactor));
        if (zoomFactor >= 2.0f) scaledFontSize -= static_cast<int>(zoomFactor);
        if (scaledFontSize < 1) scaledFontSize = 1;
        return scaledFontSize;
    }

    //! Returns perspective mode
    static bool GetPerspectiveState() {
        assert(GetInstance());
        return GetInstance()->getPerspectiveState();
    }
    inline virtual bool getPerspectiveState(int viewport = 0) const {
        return false;
    }

    template <typename iType, typename oType>
    static void ToWorldPoint(const Vector3Tpl<iType>& input2D,
                             Vector3Tpl<oType>& output3D) {
        GetInstance()->toWorldPoint(input2D, output3D);
    }

    inline virtual void toWorldPoint(const CCVector3d& input2D,
                                     CCVector3d& output3D) { /* do nothing */ }
    // inline static void ToWorldPoint(CCVector3 & p) {
    // s_tools->toWorldPoint(p); }
    inline virtual void toWorldPoint(const CCVector3& input2D,
                                     CCVector3d& output3D) { /* do nothing */ }

    template <typename iType, typename oType>
    inline static void ToDisplayPoint(const Vector3Tpl<iType>& input3D,
                                      Vector3Tpl<oType>& output2D) {
        GetInstance()->toDisplayPoint(input3D, output2D);
    }
    inline virtual void toDisplayPoint(const CCVector3d& input3D,
                                       CCVector3d& output2D) { /* do nothing */
    }
    // inline static void ToDisplayPoint(const CCVector3 & worldPos, CCVector3d
    // & displayPos) { s_tools->toDisplayPoint(worldPos, displayPos); }
    inline virtual void toDisplayPoint(const CCVector3& input3D,
                                       CCVector3d& output2D) { /* do nothing */
    }

    //! Text alignment
    enum TextAlign {
        ALIGN_HLEFT = 1,
        ALIGN_HMIDDLE = 2,
        ALIGN_HRIGHT = 4,
        ALIGN_VTOP = 8,
        ALIGN_VMIDDLE = 16,
        ALIGN_VBOTTOM = 32,
        ALIGN_DEFAULT = 1 | 8
    };

public:  // GLU equivalent methods
    static ccGLMatrixd Frustum(double left,
                               double right,
                               double bottom,
                               double top,
                               double znear,
                               double zfar) {
        // invalid for: n<=0, f<=0, l=r, b=t, or n=f
        assert(znear > 0);
        assert(zfar > 0);
        assert(left != right);
        assert(bottom != top);
        assert(znear != zfar);

        ccGLMatrixd outMatrix;
        {
            double* matrix = outMatrix.data();

            double dX = right - left;
            double dY = top - bottom;
            double dZ = znear - zfar;

            matrix[0] = 2 * znear / dX;
            matrix[1] = 0.0;
            matrix[2] = 0.0;
            matrix[3] = 0.0;

            matrix[4] = 0.0;
            matrix[5] = 2 * znear / dY;
            matrix[6] = 0.0;
            matrix[7] = 0.0;

            matrix[8] = (right + left) / dX;
            matrix[9] = (top + bottom) / dY;
            matrix[10] = (zfar + znear) / dZ;
            matrix[11] = -1.0;

            matrix[12] = 0.0;
            matrix[13] = 0.0;
            matrix[14] = 2 * znear * zfar / dZ;
            matrix[15] = 0.0;
        }

        return outMatrix;
    }

    // inspired from https://www.opengl.org/wiki/GluPerspective_code and
    // http://www.songho.ca/opengl/gl_projectionmatrix.html
    static ccGLMatrixd Perspective(double fovyInDegrees,
                                   double aspectRatio,
                                   double znear,
                                   double zfar) {
        ccGLMatrixd outMatrix;
        {
            double* matrix = outMatrix.data();

            double ymax =
                    znear *
                    std::tan(cloudViewer::DegreesToRadians(fovyInDegrees / 2));
            double xmax = ymax * aspectRatio;

            double dZ = zfar - znear;
            matrix[0] = znear / xmax;
            matrix[1] = 0.0;
            matrix[2] = 0.0;
            matrix[3] = 0.0;

            matrix[4] = 0.0;
            matrix[5] = znear / ymax;
            matrix[6] = 0.0;
            matrix[7] = 0.0;

            matrix[8] = 0.0;
            matrix[9] = 0.0;
            matrix[10] = -(zfar + znear) / dZ;
            matrix[11] = -1.0;

            matrix[12] = 0.0;
            matrix[13] = 0.0;
            matrix[14] = -(2.0 * znear * zfar) / dZ;
            matrix[15] = 0.0;
        }

        return outMatrix;
    }

    static ccGLMatrixd Ortho(double left,
                             double right,
                             double bottom,
                             double top,
                             double nearVal,
                             double farVal) {
        ccGLMatrixd matrix;
        double dx = (right - left);
        double dy = (top - bottom);
        double dz = (farVal - nearVal);
        if (dx != 0 && dy != 0 && dz != 0) {
            double* mat = matrix.data();
            // set OpenGL perspective projection matrix
            mat[0] = 2.0 / dx;
            mat[1] = 0;
            mat[2] = 0;
            mat[3] = 0;

            mat[4] = 0;
            mat[5] = 2.0 / dy;
            mat[6] = 0;
            mat[7] = 0;

            mat[8] = 0;
            mat[9] = 0;
            mat[10] = -2.0 / dz;
            mat[11] = 0;

            mat[12] = -(right + left) / dx;
            mat[13] = -(top + bottom) / dy;
            mat[14] = -(farVal + nearVal) / dz;
            mat[15] = 1.0;
        } else {
            matrix.toIdentity();
        }

        return matrix;
    }

    // inspired from http://www.songho.ca/opengl/gl_projectionmatrix.html
    static ccGLMatrixd Ortho(double w, double h, double d) {
        ccGLMatrixd matrix;
        if (w != 0 && h != 0 && d != 0) {
            double* mat = matrix.data();
            mat[0] = 1.0 / w;
            mat[1] = 0.0;
            mat[2] = 0.0;
            mat[3] = 0.0;

            mat[4] = 0.0;
            mat[5] = 1.0 / h;
            mat[6] = 0.0;
            mat[7] = 0.0;

            mat[8] = 0.0;
            mat[9] = 0.0;
            mat[10] = -1.0 / d;
            mat[11] = 0.0;

            mat[12] = 0.0;
            mat[13] = 0.0;
            mat[14] = 0.0;
            mat[15] = 1.0;
        } else {
            matrix.toIdentity();
        }

        return matrix;
    }

    template <typename iType, typename oType>
    static bool Project(const Vector3Tpl<iType>& input3D,
                        const oType* modelview,
                        const oType* projection,
                        const int* viewport,
                        Vector3Tpl<oType>& output2D,
                        bool* inFrustum = nullptr) {
        if (GetInstance() && !GetPerspectiveState()) {
            ToDisplayPoint<iType, oType>(input3D, output2D);
            return true;
        }

        // Modelview transform
        Tuple4Tpl<oType> Pm;
        {
            Pm.x = static_cast<oType>(modelview[0] * input3D.x +
                                      modelview[4] * input3D.y +
                                      modelview[8] * input3D.z + modelview[12]);
            Pm.y = static_cast<oType>(modelview[1] * input3D.x +
                                      modelview[5] * input3D.y +
                                      modelview[9] * input3D.z + modelview[13]);
            Pm.z = static_cast<oType>(
                    modelview[2] * input3D.x + modelview[6] * input3D.y +
                    modelview[10] * input3D.z + modelview[14]);
            Pm.w = static_cast<oType>(
                    modelview[3] * input3D.x + modelview[7] * input3D.y +
                    modelview[11] * input3D.z + modelview[15]);
        };

        // Projection transform
        Tuple4Tpl<oType> Pp;
        {
            Pp.x = static_cast<oType>(
                    projection[0] * Pm.x + projection[4] * Pm.y +
                    projection[8] * Pm.z + projection[12] * Pm.w);
            Pp.y = static_cast<oType>(
                    projection[1] * Pm.x + projection[5] * Pm.y +
                    projection[9] * Pm.z + projection[13] * Pm.w);
            Pp.z = static_cast<oType>(
                    projection[2] * Pm.x + projection[6] * Pm.y +
                    projection[10] * Pm.z + projection[14] * Pm.w);
            Pp.w = static_cast<oType>(
                    projection[3] * Pm.x + projection[7] * Pm.y +
                    projection[11] * Pm.z + projection[15] * Pm.w);
        };

        // The result normalizes between -1 and 1
        if (Pp.w == 0.0) {
            return false;
        }

        if (inFrustum) {
            // Check if the point is inside the frustum
            *inFrustum = (std::abs(Pp.x) <= Pp.w && std::abs(Pp.y) <= Pp.w &&
                          std::abs(Pp.z) <= Pp.w);
        }

        // Perspective division
        Pp.x /= Pp.w;
        Pp.y /= Pp.w;
        Pp.z /= Pp.w;
        // Window coordinates
        // Map x, y to range 0-1
        output2D.x = (1.0 + Pp.x) / 2 * viewport[2] + viewport[0];
        output2D.y = (1.0 + Pp.y) / 2 * viewport[3] + viewport[1];
        // This is only correct when glDepthRange(0.0, 1.0)
        output2D.z = (1.0 + Pp.z) / 2;  // Between 0 and 1

        return true;
    }

    inline static double MAT(const double* m, int r, int c) {
        return m[c * 4 + r];
    }
    inline static float MAT(const float* m, int r, int c) {
        return m[c * 4 + r];
    }

    inline static double& MAT(double* m, int r, int c) { return m[c * 4 + r]; }
    inline static float& MAT(float* m, int r, int c) { return m[c * 4 + r]; }

    template <typename Type>
    static bool InvertMatrix(const Type* m, Type* out) {
        Type wtmp[4][8];
        Type m0, m1, m2, m3, s;
        Type *r0, *r1, *r2, *r3;
        r0 = wtmp[0], r1 = wtmp[1], r2 = wtmp[2], r3 = wtmp[3];

        r0[0] = MAT(m, 0, 0), r0[1] = MAT(m, 0, 1), r0[2] = MAT(m, 0, 2),
        r0[3] = MAT(m, 0, 3), r0[4] = 1.0, r0[5] = r0[6] = r0[7] = 0.0,
        r1[0] = MAT(m, 1, 0), r1[1] = MAT(m, 1, 1), r1[2] = MAT(m, 1, 2),
        r1[3] = MAT(m, 1, 3), r1[5] = 1.0, r1[4] = r1[6] = r1[7] = 0.0,
        r2[0] = MAT(m, 2, 0), r2[1] = MAT(m, 2, 1), r2[2] = MAT(m, 2, 2),
        r2[3] = MAT(m, 2, 3), r2[6] = 1.0, r2[4] = r2[5] = r2[7] = 0.0,
        r3[0] = MAT(m, 3, 0), r3[1] = MAT(m, 3, 1), r3[2] = MAT(m, 3, 2),
        r3[3] = MAT(m, 3, 3), r3[7] = 1.0, r3[4] = r3[5] = r3[6] = 0.0;

        // choose pivot - or die
        if (std::abs(r3[0]) > std::abs(r2[0])) std::swap(r3, r2);
        if (std::abs(r2[0]) > std::abs(r1[0])) std::swap(r2, r1);
        if (std::abs(r1[0]) > std::abs(r0[0])) std::swap(r1, r0);
        if (0.0 == r0[0]) return false;

        // eliminate first variable
        m1 = r1[0] / r0[0];
        m2 = r2[0] / r0[0];
        m3 = r3[0] / r0[0];
        s = r0[1];
        r1[1] -= m1 * s;
        r2[1] -= m2 * s;
        r3[1] -= m3 * s;
        s = r0[2];
        r1[2] -= m1 * s;
        r2[2] -= m2 * s;
        r3[2] -= m3 * s;
        s = r0[3];
        r1[3] -= m1 * s;
        r2[3] -= m2 * s;
        r3[3] -= m3 * s;
        s = r0[4];
        if (s != 0.0) {
            r1[4] -= m1 * s;
            r2[4] -= m2 * s;
            r3[4] -= m3 * s;
        }
        s = r0[5];
        if (s != 0.0) {
            r1[5] -= m1 * s;
            r2[5] -= m2 * s;
            r3[5] -= m3 * s;
        }
        s = r0[6];
        if (s != 0.0) {
            r1[6] -= m1 * s;
            r2[6] -= m2 * s;
            r3[6] -= m3 * s;
        }
        s = r0[7];
        if (s != 0.0) {
            r1[7] -= m1 * s;
            r2[7] -= m2 * s;
            r3[7] -= m3 * s;
        }

        // choose pivot - or die
        if (std::abs(r3[1]) > std::abs(r2[1])) std::swap(r3, r2);
        if (std::abs(r2[1]) > std::abs(r1[1])) std::swap(r2, r1);
        if (0.0 == r1[1]) return false;

        // eliminate second variable
        m2 = r2[1] / r1[1];
        m3 = r3[1] / r1[1];
        r2[2] -= m2 * r1[2];
        r3[2] -= m3 * r1[2];
        r2[3] -= m2 * r1[3];
        r3[3] -= m3 * r1[3];
        s = r1[4];
        if (0.0 != s) {
            r2[4] -= m2 * s;
            r3[4] -= m3 * s;
        }
        s = r1[5];
        if (0.0 != s) {
            r2[5] -= m2 * s;
            r3[5] -= m3 * s;
        }
        s = r1[6];
        if (0.0 != s) {
            r2[6] -= m2 * s;
            r3[6] -= m3 * s;
        }
        s = r1[7];
        if (0.0 != s) {
            r2[7] -= m2 * s;
            r3[7] -= m3 * s;
        }

        // choose pivot - or die
        if (std::abs(r3[2]) > std::abs(r2[2])) std::swap(r3, r2);
        if (0.0 == r2[2]) return false;

        // eliminate third variable
        m3 = r3[2] / r2[2];
        r3[3] -= m3 * r2[3], r3[4] -= m3 * r2[4], r3[5] -= m3 * r2[5],
                r3[6] -= m3 * r2[6], r3[7] -= m3 * r2[7];

        // last check
        if (0.0 == r3[3]) return false;

        s = 1.0 / r3[3];  // now back substitute row 3
        r3[4] *= s;
        r3[5] *= s;
        r3[6] *= s;
        r3[7] *= s;
        m2 = r2[3];  // now back substitute row 2
        s = 1.0 / r2[2];
        r2[4] = s * (r2[4] - r3[4] * m2), r2[5] = s * (r2[5] - r3[5] * m2),
        r2[6] = s * (r2[6] - r3[6] * m2), r2[7] = s * (r2[7] - r3[7] * m2);
        m1 = r1[3];
        r1[4] -= r3[4] * m1, r1[5] -= r3[5] * m1, r1[6] -= r3[6] * m1,
                r1[7] -= r3[7] * m1;
        m0 = r0[3];
        r0[4] -= r3[4] * m0, r0[5] -= r3[5] * m0, r0[6] -= r3[6] * m0,
                r0[7] -= r3[7] * m0;
        m1 = r1[2];  // now back substitute row 1
        s = 1.0 / r1[1];
        r1[4] = s * (r1[4] - r2[4] * m1), r1[5] = s * (r1[5] - r2[5] * m1),
        r1[6] = s * (r1[6] - r2[6] * m1), r1[7] = s * (r1[7] - r2[7] * m1);
        m0 = r0[2];
        r0[4] -= r2[4] * m0, r0[5] -= r2[5] * m0, r0[6] -= r2[6] * m0,
                r0[7] -= r2[7] * m0;
        m0 = r0[1];  // now back substitute row 0
        s = 1.0 / r0[0];
        r0[4] = s * (r0[4] - r1[4] * m0), r0[5] = s * (r0[5] - r1[5] * m0),
        r0[6] = s * (r0[6] - r1[6] * m0), r0[7] = s * (r0[7] - r1[7] * m0);

        MAT(out, 0, 0) = r0[4];
        MAT(out, 0, 1) = r0[5], MAT(out, 0, 2) = r0[6];
        MAT(out, 0, 3) = r0[7], MAT(out, 1, 0) = r1[4];
        MAT(out, 1, 1) = r1[5], MAT(out, 1, 2) = r1[6];
        MAT(out, 1, 3) = r1[7], MAT(out, 2, 0) = r2[4];
        MAT(out, 2, 1) = r2[5], MAT(out, 2, 2) = r2[6];
        MAT(out, 2, 3) = r2[7], MAT(out, 3, 0) = r3[4];
        MAT(out, 3, 1) = r3[5], MAT(out, 3, 2) = r3[6];
        MAT(out, 3, 3) = r3[7];

        return true;
    }

    template <typename iType, typename oType>
    static bool Unproject(const Vector3Tpl<iType>& input2D,
                          const oType* modelview,
                          const oType* projection,
                          const int* viewport,
                          Vector3Tpl<oType>& output3D) {
        if (GetInstance() && !GetPerspectiveState()) {
            ToWorldPoint<iType, oType>(input2D, output3D);
            return true;
        }

        // compute projection x modelview
        ccGLMatrixTpl<oType> A = ccGLMatrixTpl<oType>(projection) *
                                 ccGLMatrixTpl<oType>(modelview);
        ccGLMatrixTpl<oType> m;

        if (!InvertMatrix(A.data(), m.data())) {
            return false;
        }

        ccGLMatrixTpl<oType> mA = m * A;

        // Transformation of normalized coordinates between -1 and 1
        Tuple4Tpl<oType> in;
        in.x = static_cast<oType>(
                (input2D.x - static_cast<iType>(viewport[0])) / viewport[2] *
                        2 -
                1);
        in.y = static_cast<oType>(
                (input2D.y - static_cast<iType>(viewport[1])) / viewport[3] *
                        2 -
                1);
        in.z = static_cast<oType>(2 * input2D.z - 1);
        in.w = 1;

        // Objects coordinates
        Tuple4Tpl<oType> out = m * in;
        if (out.w == 0) {
            return false;
        }

        output3D = Vector3Tpl<oType>(out.u) / out.w;

        return true;
    }

    static void PickMatrix(double x,
                           double y,
                           double width,
                           double height,
                           int viewport[4],
                           double m[16]) {
        double sx = viewport[2] / width;
        double sy = viewport[3] / height;
        double tx = (viewport[2] + 2.0 * (viewport[0] - x)) / width;
        double ty = (viewport[3] + 2.0 * (viewport[1] - y)) / height;

        MAT(m, 0, 0) = sx;
        MAT(m, 0, 1) = 0.0;
        MAT(m, 0, 2) = 0.0;
        MAT(m, 0, 3) = tx;
        MAT(m, 1, 0) = 0.0;
        MAT(m, 1, 1) = sy;
        MAT(m, 1, 2) = 0.0;
        MAT(m, 1, 3) = ty;
        MAT(m, 2, 0) = 0.0;
        MAT(m, 2, 1) = 0.0;
        MAT(m, 2, 2) = 1.0;
        MAT(m, 2, 3) = 0.0;
        MAT(m, 3, 0) = 0.0;
        MAT(m, 3, 1) = 0.0;
        MAT(m, 3, 2) = 0.0;
        MAT(m, 3, 3) = 1.0;
    }

protected:
    //! Default constructor
    ecvGenericDisplayTools();
};

//! OpenGL camera parameters
struct ECV_DB_LIB_API ccGLCameraParameters {
    ccGLCameraParameters()
        : perspective(false), fov_deg(0.0f), pixelSize(0.0f) {
        memset(viewport, 0, 4 * sizeof(int));
    }

    //! Projects a 3D point in 2D (+ normalized 'z' coordinate)
    inline bool project(const CCVector3d& input3D,
                        CCVector3d& output2D,
                        bool* inFrustum = nullptr) const {
        return ecvGenericDisplayTools::Project<double, double>(
                input3D, modelViewMat.data(), projectionMat.data(), viewport,
                output2D, inFrustum);
    }
    //! Projects a 3D point in 2D (+ normalized 'z' coordinate)
    inline bool project(const CCVector3& input3D,
                        CCVector3d& output2D,
                        bool* inFrustum = nullptr) const {
        return ecvGenericDisplayTools::Project<PointCoordinateType, double>(
                input3D, modelViewMat.data(), projectionMat.data(), viewport,
                output2D, inFrustum);
    }

    //! Unprojects a 2D point (+ normalized 'z' coordinate) in 3D
    inline bool unproject(const CCVector3d& input2D,
                          CCVector3d& output3D) const {
        return ecvGenericDisplayTools::Unproject<double, double>(
                input2D, modelViewMat.data(), projectionMat.data(), viewport,
                output3D);
    }
    //! Unprojects a 2D point (+ normalized 'z' coordinate) in 3D
    inline bool unproject(const CCVector3& input2D,
                          CCVector3d& output3D) const {
        return ecvGenericDisplayTools::Unproject<PointCoordinateType, double>(
                input2D, modelViewMat.data(), projectionMat.data(), viewport,
                output3D);
    }

    //! Model view matrix (GL_MODELVIEW)
    ccGLMatrixd modelViewMat;
    //! Projection matrix (GL_PROJECTION)
    ccGLMatrixd projectionMat;
    //! Viewport (GL_VIEWPORT)
    int viewport[4];
    //! Perspective mode
    bool perspective;
    //! F.O.V. (in degrees) - perspective mode only
    float fov_deg;
    //! Pixel size (i.e. zoom) - non perspective mode only
    float pixelSize;
};
