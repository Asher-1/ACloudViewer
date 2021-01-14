// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "visualization/shader/SimpleShader.h"

#include <GenericIndexedMesh.h>

#include <ecvBBox.h>
#include <ecvOrientedBBox.h>
#include <LineSet.h>
#include <Octree.h>
#include <ecvCone.h>
#include <ecvAdvancedTypes.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvHalfEdgeMesh.h>
#include <VoxelGrid.h>
#include <ecvPolyline.h>
#include <ecvTetraMesh.h>
#include <ecvPointCloud.h>
#include "visualization/shader/Shader.h"
#include "visualization/utility/ColorMap.h"

// SYSTEM
#include <unordered_set>

namespace cloudViewer {
namespace visualization {
namespace glsl {
	using namespace CVLib;

// Coordinates of 8 vertices in a cuboid (assume origin (0,0,0), size 1)
const static std::vector<Eigen::Vector3i> cuboid_vertex_offsets{
        Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(1, 0, 0),
        Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(1, 1, 0),
        Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(1, 0, 1),
        Eigen::Vector3i(0, 1, 1), Eigen::Vector3i(1, 1, 1),
};

// Vertex indices of 12 triangles in a cuboid, for right-handed manifold mesh
const static std::vector<Eigen::Vector3i> cuboid_triangles_vertex_indices{
        Eigen::Vector3i(0, 2, 1), Eigen::Vector3i(0, 1, 4),
        Eigen::Vector3i(0, 4, 2), Eigen::Vector3i(5, 1, 7),
        Eigen::Vector3i(5, 7, 4), Eigen::Vector3i(5, 4, 1),
        Eigen::Vector3i(3, 7, 1), Eigen::Vector3i(3, 1, 2),
        Eigen::Vector3i(3, 2, 7), Eigen::Vector3i(6, 4, 7),
        Eigen::Vector3i(6, 7, 2), Eigen::Vector3i(6, 2, 4),
};

// Vertex indices of 12 lines in a cuboid
const static std::vector<Eigen::Vector2i> cuboid_lines_vertex_indices{
        Eigen::Vector2i(0, 1), Eigen::Vector2i(0, 2), Eigen::Vector2i(0, 4),
        Eigen::Vector2i(3, 1), Eigen::Vector2i(3, 2), Eigen::Vector2i(3, 7),
        Eigen::Vector2i(5, 1), Eigen::Vector2i(5, 4), Eigen::Vector2i(5, 7),
        Eigen::Vector2i(6, 2), Eigen::Vector2i(6, 4), Eigen::Vector2i(6, 7),
};

//! Shortcuts to OpenGL commands independent on the input type
class GLUtility
{
public:

	//type-less glVertex3Xv call (X=f,d)
	static inline void Vertex3v(const float* v) { glVertex3fv(v); }
	static inline void Vertex3v(const double* v) { glVertex3dv(v); }

	//type-less glVertex3X call (X=f,d)
	static inline void Vertex3(float x, float y, float z) { glVertex3f(x, y, z); }
	static inline void Vertex3(double x, double y, double z) { glVertex3d(x, y, z); }

	//type-less glScaleX call (X=f,d)
	static inline void Scale(float x, float y, float z) { glScalef(x, y, z); }
	static inline void Scale(double x, double y, double z) { glScaled(x, y, z); }

	//type-less glNormal3Xv call (X=f,d)
	static inline void Normal3v(const float* v) { glNormal3fv(v); }
	static inline void Normal3v(const double* v) { glNormal3dv(v); }

	//type-less glRotateX call (X=f,d)
	static inline void Rotate(float a, float x, float y, float z) { glRotatef(a, x, y, z); }
	static inline void Rotate(double a, double x, double y, double z) { glRotated(a, x, y, z); }

	//type-less glTranslateX call (X=f,d)
	static inline void Translate(float x, float y, float z) { glTranslatef(x, y, z); }
	static inline void Translate(double x, double y, double z) { glTranslated(x, y, z); }

	//type-less glColor3Xv call (X=f,ub)
	static inline void Color3v(const unsigned char* v) { glColor3ubv(v); }
	static inline void Color3v(const float* v) { glColor3fv(v); }

	//type-less glColor4Xv call (X=f,ub)
	static inline void Color4v(const unsigned char* v) { glColor4ubv(v); }
	static inline void Color4v(const float* v) { glColor4fv(v); }

public: //GLU equivalent methods

	static ccGLMatrixd Frustum(double left, double right, double bottom, double top, double znear, double zfar)
	{
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
			matrix[14] = 2 * znear*zfar / dZ;
			matrix[15] = 0.0;
		}

		return outMatrix;
	}

	//inspired from https://www.opengl.org/wiki/GluPerspective_code and http://www.songho.ca/opengl/gl_projectionmatrix.html
	static ccGLMatrixd Perspective(double fovyInDegrees, double aspectRatio, double znear, double zfar)
	{
		ccGLMatrixd outMatrix;
		{
			double* matrix = outMatrix.data();

			double ymax = znear * std::tan(fovyInDegrees / 2 * CV_DEG_TO_RAD);
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

	//inspired from http://www.songho.ca/opengl/gl_projectionmatrix.html
	static ccGLMatrixd Ortho(double w, double h, double d)
	{
		ccGLMatrixd matrix;
		if (w != 0 && h != 0 && d != 0)
		{
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
		}
		else
		{
			matrix.toIdentity();
		}

		return matrix;
	}

	template <typename iType, typename oType>
	static bool Project(const Vector3Tpl<iType>& input3D, const oType* modelview, const oType* projection, const int* viewport, Vector3Tpl<oType>& output2D, bool* inFrustum = nullptr)
	{
		//Modelview transform
		Tuple4Tpl<oType> Pm;
		{
			Pm.x = static_cast<oType>(modelview[0] * input3D.x + modelview[4] * input3D.y + modelview[8] * input3D.z + modelview[12]);
			Pm.y = static_cast<oType>(modelview[1] * input3D.x + modelview[5] * input3D.y + modelview[9] * input3D.z + modelview[13]);
			Pm.z = static_cast<oType>(modelview[2] * input3D.x + modelview[6] * input3D.y + modelview[10] * input3D.z + modelview[14]);
			Pm.w = static_cast<oType>(modelview[3] * input3D.x + modelview[7] * input3D.y + modelview[11] * input3D.z + modelview[15]);
		};

		//Projection transform
		Tuple4Tpl<oType> Pp;
		{
			Pp.x = static_cast<oType>(projection[0] * Pm.x + projection[4] * Pm.y + projection[8] * Pm.z + projection[12] * Pm.w);
			Pp.y = static_cast<oType>(projection[1] * Pm.x + projection[5] * Pm.y + projection[9] * Pm.z + projection[13] * Pm.w);
			Pp.z = static_cast<oType>(projection[2] * Pm.x + projection[6] * Pm.y + projection[10] * Pm.z + projection[14] * Pm.w);
			Pp.w = static_cast<oType>(projection[3] * Pm.x + projection[7] * Pm.y + projection[11] * Pm.z + projection[15] * Pm.w);
		};

		//The result normalizes between -1 and 1
		if (Pp.w == 0.0)
		{
			return false;
		}

		if (inFrustum)
		{
			//Check if the point is inside the frustum
			*inFrustum = (std::abs(Pp.x) <= Pp.w && std::abs(Pp.y) <= Pp.w && std::abs(Pp.z) <= Pp.w);
		}

		//Perspective division
		Pp.x /= Pp.w;
		Pp.y /= Pp.w;
		Pp.z /= Pp.w;
		//Window coordinates
		//Map x, y to range 0-1
		output2D.x = (1.0 + Pp.x) / 2 * viewport[2] + viewport[0];
		output2D.y = (1.0 + Pp.y) / 2 * viewport[3] + viewport[1];
		//This is only correct when glDepthRange(0.0, 1.0)
		output2D.z = (1.0 + Pp.z) / 2;	//Between 0 and 1

		return true;
	}

	inline static double MAT(const double* m, int r, int c) { return m[c * 4 + r]; }
	inline static float MAT(const float* m, int r, int c) { return m[c * 4 + r]; }

	inline static double& MAT(double* m, int r, int c) { return m[c * 4 + r]; }
	inline static float& MAT(float* m, int r, int c) { return m[c * 4 + r]; }

	template <typename Type>
	static bool InvertMatrix(const Type* m, Type* out)
	{
		Type wtmp[4][8];
		Type m0, m1, m2, m3, s;
		Type *r0, *r1, *r2, *r3;
		r0 = wtmp[0], r1 = wtmp[1], r2 = wtmp[2], r3 = wtmp[3];

		r0[0] = MAT(m, 0, 0), r0[1] = MAT(m, 0, 1),
			r0[2] = MAT(m, 0, 2), r0[3] = MAT(m, 0, 3),
			r0[4] = 1.0, r0[5] = r0[6] = r0[7] = 0.0,
			r1[0] = MAT(m, 1, 0), r1[1] = MAT(m, 1, 1),
			r1[2] = MAT(m, 1, 2), r1[3] = MAT(m, 1, 3),
			r1[5] = 1.0, r1[4] = r1[6] = r1[7] = 0.0,
			r2[0] = MAT(m, 2, 0), r2[1] = MAT(m, 2, 1),
			r2[2] = MAT(m, 2, 2), r2[3] = MAT(m, 2, 3),
			r2[6] = 1.0, r2[4] = r2[5] = r2[7] = 0.0,
			r3[0] = MAT(m, 3, 0), r3[1] = MAT(m, 3, 1),
			r3[2] = MAT(m, 3, 2), r3[3] = MAT(m, 3, 3),
			r3[7] = 1.0, r3[4] = r3[5] = r3[6] = 0.0;

		//choose pivot - or die
		if (std::abs(r3[0]) > std::abs(r2[0]))
			std::swap(r3, r2);
		if (std::abs(r2[0]) > std::abs(r1[0]))
			std::swap(r2, r1);
		if (std::abs(r1[0]) > std::abs(r0[0]))
			std::swap(r1, r0);
		if (0.0 == r0[0])
			return false;

		//eliminate first variable
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
		if (s != 0.0)
		{
			r1[4] -= m1 * s;
			r2[4] -= m2 * s;
			r3[4] -= m3 * s;
		}
		s = r0[5];
		if (s != 0.0)
		{
			r1[5] -= m1 * s;
			r2[5] -= m2 * s;
			r3[5] -= m3 * s;
		}
		s = r0[6];
		if (s != 0.0)
		{
			r1[6] -= m1 * s;
			r2[6] -= m2 * s;
			r3[6] -= m3 * s;
		}
		s = r0[7];
		if (s != 0.0)
		{
			r1[7] -= m1 * s;
			r2[7] -= m2 * s;
			r3[7] -= m3 * s;
		}

		//choose pivot - or die
		if (std::abs(r3[1]) > std::abs(r2[1]))
			std::swap(r3, r2);
		if (std::abs(r2[1]) > std::abs(r1[1]))
			std::swap(r2, r1);
		if (0.0 == r1[1])
			return false;

		//eliminate second variable
		m2 = r2[1] / r1[1];
		m3 = r3[1] / r1[1];
		r2[2] -= m2 * r1[2];
		r3[2] -= m3 * r1[2];
		r2[3] -= m2 * r1[3];
		r3[3] -= m3 * r1[3];
		s = r1[4];
		if (0.0 != s)
		{
			r2[4] -= m2 * s;
			r3[4] -= m3 * s;
		}
		s = r1[5];
		if (0.0 != s)
		{
			r2[5] -= m2 * s;
			r3[5] -= m3 * s;
		}
		s = r1[6];
		if (0.0 != s)
		{
			r2[6] -= m2 * s;
			r3[6] -= m3 * s;
		}
		s = r1[7];
		if (0.0 != s)
		{
			r2[7] -= m2 * s;
			r3[7] -= m3 * s;
		}

		//choose pivot - or die
		if (std::abs(r3[2]) > std::abs(r2[2]))
			std::swap(r3, r2);
		if (0.0 == r2[2])
			return false;

		//eliminate third variable
		m3 = r3[2] / r2[2];
		r3[3] -= m3 * r2[3], r3[4] -= m3 * r2[4],
			r3[5] -= m3 * r2[5], r3[6] -= m3 * r2[6], r3[7] -= m3 * r2[7];

		//last check
		if (0.0 == r3[3])
			return false;

		s = 1.0 / r3[3]; //now back substitute row 3
		r3[4] *= s;
		r3[5] *= s;
		r3[6] *= s;
		r3[7] *= s;
		m2 = r2[3]; //now back substitute row 2
		s = 1.0 / r2[2];
		r2[4] = s * (r2[4] - r3[4] * m2), r2[5] = s * (r2[5] - r3[5] * m2),
			r2[6] = s * (r2[6] - r3[6] * m2), r2[7] = s * (r2[7] - r3[7] * m2);
		m1 = r1[3];
		r1[4] -= r3[4] * m1, r1[5] -= r3[5] * m1,
			r1[6] -= r3[6] * m1, r1[7] -= r3[7] * m1;
		m0 = r0[3];
		r0[4] -= r3[4] * m0, r0[5] -= r3[5] * m0,
			r0[6] -= r3[6] * m0, r0[7] -= r3[7] * m0;
		m1 = r1[2]; //now back substitute row 1
		s = 1.0 / r1[1];
		r1[4] = s * (r1[4] - r2[4] * m1), r1[5] = s * (r1[5] - r2[5] * m1),
			r1[6] = s * (r1[6] - r2[6] * m1), r1[7] = s * (r1[7] - r2[7] * m1);
		m0 = r0[2];
		r0[4] -= r2[4] * m0, r0[5] -= r2[5] * m0,
			r0[6] -= r2[6] * m0, r0[7] -= r2[7] * m0;
		m0 = r0[1]; //now back substitute row 0
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
	static bool Unproject(const Vector3Tpl<iType>& input2D, const oType* modelview, const oType* projection, const int* viewport, Vector3Tpl<oType>& output3D)
	{
		//compute projection x modelview
		ccGLMatrixTpl<oType> A = ccGLMatrixTpl<oType>(projection) * ccGLMatrixTpl<oType>(modelview);
		ccGLMatrixTpl<oType> m;

		if (!InvertMatrix(A.data(), m.data()))
		{
			return false;
		}

		ccGLMatrixTpl<oType> mA = m * A;

		//Transformation of normalized coordinates between -1 and 1
		Tuple4Tpl<oType> in;
		in.x = static_cast<oType>((input2D.x - static_cast<iType>(viewport[0])) / viewport[2] * 2 - 1);
		in.y = static_cast<oType>((input2D.y - static_cast<iType>(viewport[1])) / viewport[3] * 2 - 1);
		in.z = static_cast<oType>(2 * input2D.z - 1);
		in.w = 1;

		//Objects coordinates
		Tuple4Tpl<oType> out = m * in;
		if (out.w == 0)
		{
			return false;
		}

		output3D = Vector3Tpl<oType>(out.u) / out.w;

		return true;
	}

	static void PickMatrix(double x, double y, double width, double height, int viewport[4], double m[16])
	{
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
};

class GLDrawUtility
{
public:
	static void DrawUtility(const ccHObject &geometry)
	{
		if (geometry.isKindOf(CV_TYPES::MESH))
		{

			glPushAttrib(GL_LIGHTING_BIT | GL_TRANSFORM_BIT | GL_ENABLE_BIT);

			glBegin(GL_TRIANGLES);
			//current vertex color (RGB)
			const ecvColor::Rgb* rgb1 = nullptr;
			const ecvColor::Rgb* rgb2 = nullptr;
			const ecvColor::Rgb* rgb3 = nullptr;

			const ccMesh& mesh = static_cast<const ccMesh&>(geometry);
			ccGenericPointCloud* genericCloud = const_cast<ccGenericPointCloud*>(mesh.getAssociatedCloud());
			ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(genericCloud);

			//display parameters
			glDrawParams glParams;
			mesh.getDrawingParameters(glParams);

			unsigned triNum = mesh.size();
			//loop on all triangles
			for (unsigned n = 0; n < triNum; ++n)
			{
				//current triangle vertices
				const CVLib::VerticesIndexes& tsi = *mesh.getTriangleVertIndexes(n);

				if (glParams.showColors)
				{
					if (mesh.isColorOverriden())
					{
						GLUtility::Color3v(mesh.getTempColor().rgb);
						glParams.showColors = false;
					} else if (cloud->hasColors())
					{
						rgb1 = &cloud->rgbColors()->at(tsi.i1);
						rgb2 = &cloud->rgbColors()->at(tsi.i2);
						rgb3 = &cloud->rgbColors()->at(tsi.i3);
					}
				}

				if (glParams.showNorms)
				{
					//if (mesh.hasTriNormals())
					//{
					//	Tuple3i idx;
					//	mesh.getTriangleNormalIndexes(n, idx.x, idx.y, idx.z);
					//	N1 = (idx.u[0] >= 0 ? ccNormalVectors::GetNormal(m_triNormals->getValue(idx.u[0])).u : nullptr);
					//	N2 = (idx.u[0] == idx.u[1] ? N1 : idx.u[1] >= 0 ? ccNormalVectors::GetNormal(m_triNormals->getValue(idx.u[1])).u : nullptr);
					//	N3 = (idx.u[0] == idx.u[2] ? N1 : idx.u[2] >= 0 ? ccNormalVectors::GetNormal(m_triNormals->getValue(idx.u[2])).u : nullptr);
					//}
					//else
					//{
					//	N1 = compressedNormals->getNormal(normalsIndexesTable->getValue(tsi.i1)).u;
					//	N2 = compressedNormals->getNormal(normalsIndexesTable->getValue(tsi.i2)).u;
					//	N3 = compressedNormals->getNormal(normalsIndexesTable->getValue(tsi.i3)).u;
					//}
				}

				//vertex 1
				//if (N1)
				//	GLUtility::Normal3v(N1);
				if (rgb1)
					GLUtility::Color3v(rgb1->rgb);
				GLUtility::Vertex3v(cloud->getPoint(tsi.i1)->u);

				//vertex 2
				//if (N2)
				//	GLUtility::Normal3v(N2);
				if (rgb2)
					GLUtility::Color3v(rgb2->rgb);
				GLUtility::Vertex3v(cloud->getPoint(tsi.i2)->u);

				//vertex 3
				//if (N3)
				//	GLUtility::Normal3v(N3);
				if (rgb3)
					GLUtility::Color3v(rgb3->rgb);
				GLUtility::Vertex3v(cloud->getPoint(tsi.i3)->u);
			}

			glEnd();
			glPopAttrib();
		}
	}
};

bool SimpleShader::Compile() {
    if (CompileShaders(SimpleVertexShader, NULL, SimpleFragmentShader) ==
        false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_color_ = glGetAttribLocation(program_, "vertex_color");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void SimpleShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool SimpleShader::BindGeometry(const ccHObject &geometry,
                                const RenderOption &option,
                                const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace InvalidateGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> colors;
    if (!PrepareBinding(geometry, option, view, points, colors)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
                 points.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &vertex_color_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f),
                 colors.data(), GL_STATIC_DRAW);
    bound_ = true;
    return true;
}

bool SimpleShader::RenderGeometry(const ccHObject &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(vertex_color_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_color_);
	if (!AdditionalRendering(geometry, option, view)) {
		PrintShaderWarning("Additional Rendering failed during preparation.");
	}
    return true;
}

void SimpleShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_color_buffer_);
        bound_ = false;
    }
}

bool SimpleShaderForPointCloud::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::POINT_CLOUD)) {
        PrintShaderWarning("Rendering type is not ccPointCloud.");
        return false;
    }
    glPointSize(GLfloat(option.point_size_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForPointCloud::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (!geometry.isKindOf(CV_TYPES::POINT_CLOUD)) {
        PrintShaderWarning("Rendering type is not ccPointCloud.");
        return false;
    }
    const ccPointCloud &pointcloud =
            (const ccPointCloud &)geometry;
    if (!pointcloud.hasPoints()) {
        PrintShaderWarning("Binding failed with empty pointcloud.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
	points.resize(pointcloud.size());
	colors.resize(pointcloud.size());
    for (size_t i = 0; i < pointcloud.size(); i++) {
        const auto &point = pointcloud.getEigenPoint(i);
        points[i] = point.cast<float>();
        Eigen::Vector3d color;
        switch (option.point_color_option_) {
            case RenderOption::PointColorOption::XCoordinate:
                color = global_color_map.GetColor(
                        view.GetBoundingBox().getXPercentage(point(0)));
                break;
            case RenderOption::PointColorOption::YCoordinate:
                color = global_color_map.GetColor(
                        view.GetBoundingBox().getYPercentage(point(1)));
                break;
            case RenderOption::PointColorOption::ZCoordinate:
                color = global_color_map.GetColor(
                        view.GetBoundingBox().getZPercentage(point(2)));
                break;
            case RenderOption::PointColorOption::Color:
            case RenderOption::PointColorOption::Default:
            default:
                if (pointcloud.isColorOverriden()) {
					color = ecvColor::Rgb::ToEigen(pointcloud.getTempColor());
				} else if (pointcloud.hasColors()) {
					color = pointcloud.getEigenColor(i);
				} else {
					color = global_color_map.GetColor(
						view.GetBoundingBox().getZPercentage(point(2)));
				}
                break;
        }
        colors[i] = color.cast<float>();
    }
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForLineSet::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::LINESET)) {
        PrintShaderWarning("Rendering type is not geometry::LineSet.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForLineSet::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (!geometry.isKindOf(CV_TYPES::LINESET)) {
        PrintShaderWarning("Rendering type is not geometry::LineSet.");
        return false;
    }
    const geometry::LineSet &lineset = (const geometry::LineSet &)geometry;
    if (!lineset.hasLines()) {
        PrintShaderWarning("Binding failed with empty geometry::LineSet.");
        return false;
    }
    points.resize(lineset.lines_.size() * 2);
    colors.resize(lineset.lines_.size() * 2);
    for (size_t i = 0; i < lineset.lines_.size(); i++) {
        const auto point_pair = lineset.getLineCoordinate(i);
        points[i * 2] = point_pair.first.cast<float>();
        points[i * 2 + 1] = point_pair.second.cast<float>();
        Eigen::Vector3d color;
        if (lineset.hasColors()) {
            color = lineset.colors_[i];
        } else {
            color = Eigen::Vector3d::Zero();
        }
        colors[i * 2] = colors[i * 2 + 1] = color.cast<float>();
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForPolyline::AdditionalRendering(
	const ccHObject & geometry, 
	const RenderOption & option, 
	const ViewControl & view)
{
	// deprecated!!!
	if (option.line_width_ != 0)
	{
		glPopAttrib();
	}
	return true;

	const ccPolyline &polyline = (const ccPolyline &)geometry;
	unsigned vertCount = polyline.size();
	unsigned arrowIndex = polyline.getArrowIndex();
	PointCoordinateType arrowLength = polyline.getArrowLength();
	if (polyline.arrowShown() && arrowIndex < vertCount &&
		(arrowIndex > 0 || polyline.isClosed()))
	{
		static std::shared_ptr<ccCone> c_unitArrow(nullptr);
		const CCVector3* P0 = polyline.getPoint(arrowIndex == 0 ? vertCount - 1 : arrowIndex - 1);
		const CCVector3* P1 = polyline.getPoint(arrowIndex);
		//direction of the last polyline chunk
		CCVector3 u = *P1 - *P0;
		u.normalize();

		if (polyline.is2DMode())
		{
			u *= -arrowLength;
			static const PointCoordinateType s_defaultArrowAngle = static_cast<PointCoordinateType>(15.0 * CV_DEG_TO_RAD);
			static const PointCoordinateType cost = cos(s_defaultArrowAngle);
			static const PointCoordinateType sint = sin(s_defaultArrowAngle);
			CCVector3 A(cost * u.x - sint * u.y, sint * u.x + cost * u.y, 0);
			CCVector3 B(cost * u.x + sint * u.y, -sint * u.x + cost * u.y, 0);
			glBegin(GL_POLYGON);
			GLUtility::Vertex3v((A + *P1).u);
			GLUtility::Vertex3v((B + *P1).u);
			GLUtility::Vertex3v((*P1).u);
			glEnd();
		}
		else
		{
			if (!c_unitArrow)
			{
				c_unitArrow = std::make_shared<ccCone>(0.5, 0.0, 1.0);
				c_unitArrow->showColors(true);
				c_unitArrow->showNormals(false);
				c_unitArrow->setVisible(true);
				c_unitArrow->setEnabled(true);
			}
			if (polyline.hasColors())
				c_unitArrow->setTempColor(polyline.getColor());
			else
				c_unitArrow->setTempColor(ecvColor::green);

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			GLUtility::Translate(P1->x, P1->y, P1->z);
			ccGLMatrix rotMat = ccGLMatrix::FromToRotation(u, CCVector3(0, 0, PC_ONE));
			glMultMatrixf(rotMat.inverse().data());
			glScalef(arrowLength, arrowLength, arrowLength);
			GLUtility::Translate(0.0, 0.0, -0.5);

			GLDrawUtility::DrawUtility(*c_unitArrow);
			glPopMatrix();
		}
	}
	
	if (option.line_width_ != 0)
	{
		glPopAttrib();
	}

	return true;
}

bool SimpleShaderForPolyline::PrepareRendering(
	const ccHObject &geometry,
	const RenderOption &option,
	const ViewControl &view) {
	if (!geometry.isKindOf(CV_TYPES::POLY_LINE)) {
		PrintShaderWarning("Rendering type is not geometry::ccPolyline.");
		return false;
	}

	if (option.line_width_ != 0)
	{
		glPushAttrib(GL_LINE_BIT);
		glLineWidth(GLfloat(option.line_width_));
	}
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GLenum(option.GetGLDepthFunc()));
	return true;
}

bool SimpleShaderForPolyline::PrepareBinding(
	const ccHObject &geometry,
	const RenderOption &option,
	const ViewControl &view,
	std::vector<Eigen::Vector3f> &points,
	std::vector<Eigen::Vector3f> &colors) {
	if (!geometry.isKindOf(CV_TYPES::POLY_LINE)) {
		PrintShaderWarning("Rendering type is not geometry::ccPolyline.");
		return false;
	}
	const ccPolyline &polyline = (const ccPolyline &)geometry;
	if (!polyline.hasPoints()) {
		PrintShaderWarning("Binding failed with empty geometry::ccPolyline.");
		return false;
	}
	points.resize((polyline.size() - 1) * 2);
	colors.resize((polyline.size() - 1) * 2);
	for (size_t i = 1; i < polyline.size(); i++) {
		points[(i-1) * 2] = CCVector3::fromArray(*polyline.getPoint(static_cast<unsigned>(i-1)));
		points[(i-1) * 2 + 1] = CCVector3::fromArray(*polyline.getPoint(static_cast<unsigned>(i)));
		Eigen::Vector3d color;
		if (polyline.isColorOverriden()) {
			color = ecvColor::Rgb::ToEigen(polyline.getTempColor());
		} else {
			color = ecvColor::Rgb::ToEigen(polyline.getColor());
		}
		colors[(i-1) * 2] = colors[(i - 1) * 2 + 1] = color.cast<float>();
	}

	if (polyline.isClosed())
	{
		if (!points.empty())
		{
			points.push_back(points[points.size() - 1]);
			points.push_back(points[0]);
		}
		
		if (!colors.empty())
		{
			colors.push_back(colors[colors.size() - 1]);
			colors.push_back(colors[0]);
		}
		
	}
	draw_arrays_mode_ = GL_LINES;
	draw_arrays_size_ = GLsizei(points.size());
	if (polyline.getWidth() > 1)
	{
		const_cast<RenderOption &>(option).line_width_ = polyline.getWidth();
	}
	return true;
}

bool SimpleShaderForTetraMesh::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::TETRA_MESH)) {
        PrintShaderWarning("Rendering type is not geometry::TetraMesh.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForTetraMesh::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    typedef decltype(geometry::TetraMesh::tetras_)::value_type TetraIndices;
    typedef decltype(geometry::TetraMesh::tetras_)::value_type::Scalar Index;
    typedef std::tuple<Index, Index> Index2;

    if (!geometry.isKindOf(CV_TYPES::TETRA_MESH)) {
        PrintShaderWarning("Rendering type is not geometry::TetraMesh.");
        return false;
    }
    const geometry::TetraMesh &tetramesh =
            (const geometry::TetraMesh &)geometry;
    if (!tetramesh.hasTetras()) {
        PrintShaderWarning("Binding failed with empty geometry::TetraMesh.");
        return false;
    }

    std::unordered_set<Index2, utility::hash_tuple::hash<Index2>>
            inserted_edges;
    auto InsertEdge = [&](Index vidx0, Index vidx1) {
        Index2 edge(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
        if (inserted_edges.count(edge) == 0) {
            inserted_edges.insert(edge);
            Eigen::Vector3f p0 = tetramesh.vertices_[vidx0].cast<float>();
            Eigen::Vector3f p1 = tetramesh.vertices_[vidx1].cast<float>();
            points.insert(points.end(), {p0, p1});
            Eigen::Vector3f color(0, 0, 0);
            colors.insert(colors.end(), {color, color});
        }
    };

    for (size_t i = 0; i < tetramesh.tetras_.size(); i++) {
        const TetraIndices tetra = tetramesh.tetras_[i];
        InsertEdge(tetra(0), tetra(1));
        InsertEdge(tetra(1), tetra(2));
        InsertEdge(tetra(2), tetra(0));
        InsertEdge(tetra(3), tetra(0));
        InsertEdge(tetra(3), tetra(1));
        InsertEdge(tetra(3), tetra(2));
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForOrientedBoundingBox::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::ORIENTED_BBOX)) {
        PrintShaderWarning(
                "Rendering type is not geometry::OrientedBoundingBox.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForOrientedBoundingBox::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (!geometry.isKindOf(CV_TYPES::ORIENTED_BBOX)) {
        PrintShaderWarning(
                "Rendering type is not geometry::OrientedBoundingBox.");
        return false;
    }
    auto lineset = geometry::LineSet::CreateFromOrientedBoundingBox(
            (const ecvOrientedBBox &)geometry);
    points.resize(lineset->lines_.size() * 2);
    colors.resize(lineset->lines_.size() * 2);
    for (size_t i = 0; i < lineset->lines_.size(); i++) {
        const auto point_pair = lineset->getLineCoordinate(i);
        points[i * 2] = point_pair.first.cast<float>();
        points[i * 2 + 1] = point_pair.second.cast<float>();
        Eigen::Vector3d color;
        if (lineset->hasColors()) {
            color = lineset->colors_[i];
        } else {
            color = Eigen::Vector3d::Zero();
        }
        colors[i * 2] = colors[i * 2 + 1] = color.cast<float>();
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForAxisAlignedBoundingBox::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::BBOX)) {
        PrintShaderWarning(
                "Rendering type is not ecvOrientedBBox.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForAxisAlignedBoundingBox::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (!geometry.isKindOf(CV_TYPES::BBOX)) {
        PrintShaderWarning(
                "Rendering type is not ccBBox.");
        return false;
    }
    auto lineset = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            (const ccBBox &)geometry);
    points.resize(lineset->lines_.size() * 2);
    colors.resize(lineset->lines_.size() * 2);
    for (size_t i = 0; i < lineset->lines_.size(); i++) {
        const auto point_pair = lineset->getLineCoordinate(i);
        points[i * 2] = point_pair.first.cast<float>();
        points[i * 2 + 1] = point_pair.second.cast<float>();
        Eigen::Vector3d color;
        if (lineset->hasColors()) {
            color = lineset->colors_[i];
        } else {
            color = Eigen::Vector3d::Zero();
        }
        colors[i * 2] = colors[i * 2 + 1] = color.cast<float>();
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForTriangleMesh::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::MESH) && 
        !geometry.isKindOf(CV_TYPES::HALF_EDGE_MESH)) {
        PrintShaderWarning("Rendering type is not ccMesh.");
        return false;
    }
    if (option.mesh_show_back_face_) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (option.mesh_show_wireframe_) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);
    } else {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    return true;
}

bool SimpleShaderForTriangleMesh::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (!geometry.isKindOf(CV_TYPES::MESH) &&
        !geometry.isKindOf(CV_TYPES::HALF_EDGE_MESH)) {
        PrintShaderWarning("Rendering type is not ccMesh.");
        return false;
    }

    if (geometry.isKindOf(CV_TYPES::MESH)) {
        const ccMesh &mesh = (const ccMesh &)geometry;
        if (!mesh.hasTriangles()) {
            PrintShaderWarning("Binding failed with empty triangle mesh.");
            return false;
        }
        const ColorMap &global_color_map = *GetGlobalColorMap();
        points.resize(mesh.size() * 3);
        colors.resize(mesh.size() * 3);

        for (unsigned int i = 0; i < mesh.size(); i++) {
            const CVLib::VerticesIndexes *triangle =
                    mesh.getTriangleVertIndexes(i);
            for (unsigned int j = 0; j < 3; j++) {
                unsigned int idx = i * 3 + j;
                unsigned int vi = triangle->i[j];
                const auto &vertex = mesh.getVertice(vi);
                points[idx] = vertex.cast<float>();

                Eigen::Vector3d color;
                switch (option.mesh_color_option_) {
                    case RenderOption::MeshColorOption::XCoordinate:
                        color = global_color_map.GetColor(
                                view.GetBoundingBox().getXPercentage(
                                        vertex(0)));
                        break;
                    case RenderOption::MeshColorOption::YCoordinate:
                        color = global_color_map.GetColor(
                                view.GetBoundingBox().getYPercentage(
                                        vertex(1)));
                        break;
                    case RenderOption::MeshColorOption::ZCoordinate:
                        color = global_color_map.GetColor(
                                view.GetBoundingBox().getZPercentage(
                                        vertex(2)));
                        break;
                    case RenderOption::MeshColorOption::Color:
                        if (mesh.isColorOverriden()) {
                            color = ecvColor::Rgb::ToEigen(mesh.getTempColor());
                            break;
                        } else if (mesh.hasColors()) {
                            color = mesh.getVertexColor(vi);
                            break;
                        }
                    case RenderOption::MeshColorOption::Default:
                    default:
                        color = option.default_mesh_color_;
                        break;
                }
                colors[idx] = color.cast<float>();
            }
        }
        draw_arrays_mode_ = GL_TRIANGLES;
        draw_arrays_size_ = GLsizei(points.size());
    } else if (geometry.isKindOf(CV_TYPES::HALF_EDGE_MESH)) {
        const geometry::ecvHalfEdgeMesh &mesh =
                (const geometry::ecvHalfEdgeMesh &)geometry;
        if (!mesh.hasTriangles()) {
            PrintShaderWarning("Binding failed with empty triangle mesh.");
            return false;
        }
        const ColorMap &global_color_map = *GetGlobalColorMap();
        points.resize(mesh.triangles_.size() * 3);
        colors.resize(mesh.triangles_.size() * 3);

        for (size_t i = 0; i < mesh.triangles_.size(); i++) {
            const auto &triangle = mesh.triangles_[i];
            for (size_t j = 0; j < 3; j++) {
                size_t idx = i * 3 + j;
                size_t vi = triangle(j);
                const auto &vertex = mesh.vertices_[vi];
                points[idx] = vertex.cast<float>();

                Eigen::Vector3d color;
                switch (option.mesh_color_option_) {
                    case RenderOption::MeshColorOption::XCoordinate:
                        color = global_color_map.GetColor(
                                view.GetBoundingBox().getXPercentage(
                                        vertex(0)));
                        break;
                    case RenderOption::MeshColorOption::YCoordinate:
                        color = global_color_map.GetColor(
                                view.GetBoundingBox().getYPercentage(
                                        vertex(1)));
                        break;
                    case RenderOption::MeshColorOption::ZCoordinate:
                        color = global_color_map.GetColor(
                                view.GetBoundingBox().getZPercentage(
                                        vertex(2)));
                        break;
                    case RenderOption::MeshColorOption::Color:
                        if (mesh.hasVertexColors()) {
                            color = mesh.vertex_colors_[vi];
                            break;
                        }
                    case RenderOption::MeshColorOption::Default:
                    default:
                        color = option.default_mesh_color_;
                        break;
                }
                colors[idx] = color.cast<float>();
            }
        }
        draw_arrays_mode_ = GL_TRIANGLES;
        draw_arrays_size_ = GLsizei(points.size());
    }

    return true;
}

bool SimpleShaderForVoxelGridLine::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::VOXEL_GRID)) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForVoxelGridLine::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (!geometry.isKindOf(CV_TYPES::VOXEL_GRID)) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    const geometry::VoxelGrid &voxel_grid =
            (const geometry::VoxelGrid &)geometry;
    if (!voxel_grid.HasVoxels()) {
        PrintShaderWarning("Binding failed with empty voxel grid.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
    points.clear();  // Final size: num_voxels * 12 * 2
    colors.clear();  // Final size: num_voxels * 12 * 2

    for (auto &it : voxel_grid.voxels_) {
        const geometry::Voxel &voxel = it.second;
        // 8 vertices in a voxel
        Eigen::Vector3f base_vertex =
                voxel_grid.origin_.cast<float>() +
                voxel.grid_index_.cast<float>() * voxel_grid.voxel_size_;
        std::vector<Eigen::Vector3f> vertices;
        for (const Eigen::Vector3i &vertex_offset : cuboid_vertex_offsets) {
            vertices.push_back(base_vertex + vertex_offset.cast<float>() *
                                                     voxel_grid.voxel_size_);
        }

        // Voxel color (applied to all points)
        Eigen::Vector3d voxel_color;
        switch (option.mesh_color_option_) {
            case RenderOption::MeshColorOption::XCoordinate:
                voxel_color = global_color_map.GetColor(
                        view.GetBoundingBox().getXPercentage(base_vertex(0)));
                break;
            case RenderOption::MeshColorOption::YCoordinate:
                voxel_color = global_color_map.GetColor(
                        view.GetBoundingBox().getYPercentage(base_vertex(1)));
                break;
            case RenderOption::MeshColorOption::ZCoordinate:
                voxel_color = global_color_map.GetColor(
                        view.GetBoundingBox().getZPercentage(base_vertex(2)));
                break;
            case RenderOption::MeshColorOption::Color:
                if (voxel_grid.HasColors()) {
                    voxel_color = voxel.color_;
                    break;
                }
            case RenderOption::MeshColorOption::Default:
            default:
                voxel_color = option.default_mesh_color_;
                break;
        }
        Eigen::Vector3f voxel_color_f = voxel_color.cast<float>();

        // 12 lines
        for (const Eigen::Vector2i &line_vertex_indices :
             cuboid_lines_vertex_indices) {
            points.push_back(vertices[line_vertex_indices(0)]);
            points.push_back(vertices[line_vertex_indices(1)]);
            colors.push_back(voxel_color_f);
            colors.push_back(voxel_color_f);
        }
    }

    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForVoxelGridFace::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::VOXEL_GRID)) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForVoxelGridFace::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (!geometry.isKindOf(CV_TYPES::VOXEL_GRID)) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    const geometry::VoxelGrid &voxel_grid =
            (const geometry::VoxelGrid &)geometry;
    if (!voxel_grid.HasVoxels()) {
        PrintShaderWarning("Binding failed with empty voxel grid.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
    points.clear();  // Final size: num_voxels * 36
    colors.clear();  // Final size: num_voxels * 36

    for (auto &it : voxel_grid.voxels_) {
        const geometry::Voxel &voxel = it.second;
        // 8 vertices in a voxel
        Eigen::Vector3f base_vertex =
                voxel_grid.origin_.cast<float>() +
                voxel.grid_index_.cast<float>() * voxel_grid.voxel_size_;
        std::vector<Eigen::Vector3f> vertices;
        for (const Eigen::Vector3i &vertex_offset : cuboid_vertex_offsets) {
            vertices.push_back(base_vertex + vertex_offset.cast<float>() *
                                                     voxel_grid.voxel_size_);
        }

        // Voxel color (applied to all points)
        Eigen::Vector3d voxel_color;
        switch (option.mesh_color_option_) {
            case RenderOption::MeshColorOption::XCoordinate:
                voxel_color = global_color_map.GetColor(
                        view.GetBoundingBox().getXPercentage(base_vertex(0)));
                break;
            case RenderOption::MeshColorOption::YCoordinate:
                voxel_color = global_color_map.GetColor(
                        view.GetBoundingBox().getYPercentage(base_vertex(1)));
                break;
            case RenderOption::MeshColorOption::ZCoordinate:
                voxel_color = global_color_map.GetColor(
                        view.GetBoundingBox().getZPercentage(base_vertex(2)));
                break;
            case RenderOption::MeshColorOption::Color:
                if (voxel_grid.HasColors()) {
                    voxel_color = voxel.color_;
                    break;
                }
            case RenderOption::MeshColorOption::Default:
            default:
                voxel_color = option.default_mesh_color_;
                break;
        }
        Eigen::Vector3f voxel_color_f = voxel_color.cast<float>();

        // 12 triangles in a voxel
        for (const Eigen::Vector3i &triangle_vertex_indices :
             cuboid_triangles_vertex_indices) {
            points.push_back(vertices[triangle_vertex_indices(0)]);
            points.push_back(vertices[triangle_vertex_indices(1)]);
            points.push_back(vertices[triangle_vertex_indices(2)]);
            colors.push_back(voxel_color_f);
            colors.push_back(voxel_color_f);
            colors.push_back(voxel_color_f);
        }
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());

    return true;
}

bool SimpleShaderForOctreeFace::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::POINT_OCTREE2)) {
        PrintShaderWarning("Rendering type is not geometry::Octree.");
        return false;
    }
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForOctreeFace::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (!geometry.isKindOf(CV_TYPES::POINT_OCTREE2)) {
        PrintShaderWarning("Rendering type is not geometry::Octree.");
        return false;
    }
    const geometry::Octree &octree = (const geometry::Octree &)geometry;
    if (octree.isEmpty()) {
        PrintShaderWarning("Binding failed with empty octree.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
    points.clear();  // Final size: num_voxels * 36
    colors.clear();  // Final size: num_voxels * 36

    auto f = [&points, &colors, &option, &global_color_map, &view](
                     const std::shared_ptr<geometry::OctreeNode> &node,
                     const std::shared_ptr<geometry::OctreeNodeInfo> &node_info)
            -> bool {
        if (auto leaf_node =
                    std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(node)) {
            // All vertex in the voxel share the same color
            Eigen::Vector3f base_vertex = node_info->origin_.cast<float>();
            std::vector<Eigen::Vector3f> vertices;
            for (const Eigen::Vector3i &vertex_offset : cuboid_vertex_offsets) {
                vertices.push_back(base_vertex +
                                   vertex_offset.cast<float>() *
                                           float(node_info->size_));
            }

            Eigen::Vector3d voxel_color;
            switch (option.mesh_color_option_) {
                case RenderOption::MeshColorOption::XCoordinate:
                    voxel_color = global_color_map.GetColor(
                            view.GetBoundingBox().getXPercentage(
                                    base_vertex(0)));
                    break;
                case RenderOption::MeshColorOption::YCoordinate:
                    voxel_color = global_color_map.GetColor(
                            view.GetBoundingBox().getYPercentage(
                                    base_vertex(1)));
                    break;
                case RenderOption::MeshColorOption::ZCoordinate:
                    voxel_color = global_color_map.GetColor(
                            view.GetBoundingBox().getZPercentage(
                                    base_vertex(2)));
                    break;
                case RenderOption::MeshColorOption::Color:
                    voxel_color = leaf_node->color_;
                    break;
                case RenderOption::MeshColorOption::Default:
                default:
                    voxel_color = option.default_mesh_color_;
                    break;
            }
            Eigen::Vector3f voxel_color_f = voxel_color.cast<float>();

            // 12 triangles in a voxel
            for (const Eigen::Vector3i &triangle_vertex_indices :
                 cuboid_triangles_vertex_indices) {
                points.push_back(vertices[triangle_vertex_indices(0)]);
                points.push_back(vertices[triangle_vertex_indices(1)]);
                points.push_back(vertices[triangle_vertex_indices(2)]);
                colors.push_back(voxel_color_f);
                colors.push_back(voxel_color_f);
                colors.push_back(voxel_color_f);
            }
        }
        return false;
    };

    octree.Traverse(f);

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());

    return true;
}

bool SimpleShaderForOctreeLine::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::POINT_OCTREE2)) {
        PrintShaderWarning("Rendering type is not geometry::Octree.");
        return false;
    }
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForOctreeLine::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (!geometry.isKindOf(CV_TYPES::POINT_OCTREE2)) {
        PrintShaderWarning("Rendering type is not geometry::Octree.");
        return false;
    }
    const geometry::Octree &octree = (const geometry::Octree &)geometry;
    if (octree.isEmpty()) {
        PrintShaderWarning("Binding failed with empty octree.");
        return false;
    }
    points.clear();  // Final size: num_voxels * 36
    colors.clear();  // Final size: num_voxels * 36

    auto f = [&points, &colors](
                     const std::shared_ptr<geometry::OctreeNode> &node,
                     const std::shared_ptr<geometry::OctreeNodeInfo> &node_info)
            -> bool {
        Eigen::Vector3f base_vertex = node_info->origin_.cast<float>();
        std::vector<Eigen::Vector3f> vertices;
        for (const Eigen::Vector3i &vertex_offset : cuboid_vertex_offsets) {
            vertices.push_back(base_vertex + vertex_offset.cast<float>() *
                                                     float(node_info->size_));
        }
        Eigen::Vector3f voxel_color = Eigen::Vector3f::Zero();
        if (auto leaf_node =
                    std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(
                            node)) {
            voxel_color = leaf_node->color_.cast<float>();
        }

        for (const Eigen::Vector2i &line_vertex_indices :
             cuboid_lines_vertex_indices) {
            points.push_back(vertices[line_vertex_indices(0)]);
            points.push_back(vertices[line_vertex_indices(1)]);
            colors.push_back(voxel_color);
            colors.push_back(voxel_color);
        }
        return false;
    };

    octree.Traverse(f);

    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());

    return true;
}

}  // namespace glsl
}  // namespace visualization
}  // namespace cloudViewer
