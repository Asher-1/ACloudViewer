//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef ECV_QUADRIC_PRIMITIVE_HEADER
#define ECV_QUADRIC_PRIMITIVE_HEADER

//Local
#include "ecvGenericPrimitive.h"

//! Quadric (primitive)
/** 2D1/2 quadric primitive
**/
class ECV_DB_LIB_API ccQuadric : public ccGenericPrimitive
{
public:

	//! Default drawing precision
	/** \warning Never pass a 'constant initializer' by reference
	**/
	static const unsigned DEFAULT_DRAWING_PRECISION = 24;

	//! Default constructor
	/** Quadric orthogonal dimension is 'Z' by default
		\param minCorner min corner of the 'representation' base area
		\param maxCorner max corner of the 'representation' base area
		\param eq equation coefficients ( Z = a + b.X + c.Y + d.X^2 + e.X.Y + f.Y^2)
		\param dims optional dimension indexes
		\param transMat optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation)
		\param name name
		\param precision drawing precision (angular step = 360/precision)
	**/
	ccQuadric(	CCVector2 minCorner,
				CCVector2 maxCorner,
				const PointCoordinateType eq[6],
				const Tuple3ub* dims = 0,
				const ccGLMatrix* transMat = 0,
				QString name = QString("Quadric"),
				unsigned precision = DEFAULT_DRAWING_PRECISION);

	//! Simplified constructor
	/** For ccHObject factory only!
	**/
	ccQuadric(QString name = QString("Quadric"));

	//! Returns class ID
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::QUADRIC; }

	//inherited from ccGenericPrimitive
	virtual QString getTypeName() const override { return "Quadric"; }
	virtual bool hasDrawingPrecision() const override { return true; }
	virtual ccGenericPrimitive* clone() const override;

	//inherited from ccHObject
	virtual ccBBox getOwnFitBB(ccGLMatrix& trans) override;

	//! Returns the quadric min corner
	const CCVector2& getMinCorner() const { return m_minCorner; }
	//! Returns the quadric max corner
	const CCVector2& getMaxCorner() const { return m_maxCorner; }

	//! Returns the quadric equation coefficients
	inline const PointCoordinateType* getEquationCoefs() const { return m_eq; }

	//! Returns the quadric equation 'coordinate system' (X,Y,Z dimensions indexes)
	inline const Tuple3ub& getEquationDims() const { return m_dims; }

	//! Projects a 3D point in the quadric coordinate system
	/** \param P input 3D point
		\param[out] Q position of the input point in the quadric coordinate system
		\return elevation of the input point (in the coordinate system quadric)
	**/
	PointCoordinateType projectOnQuadric(const CCVector3& P, CCVector3& Q) const;

	//! Returns the quadric equation coefficients as a string
	QString getEquationString() const;

	//! Fits a quadric primitive on a cloud
	/** The cloud can be any cloudViewer::GenericIndexedCloudPersist-derived object.
		\param[in] cloud input cloud
		\param[out] rms quadric fitting rms (optional)
		\return quadric primitive (if successful)
	**/
	static ccQuadric* Fit(cloudViewer::GenericIndexedCloudPersist * cloud, double* rms/*=0*/);

protected:

	//inherited from ccGenericPrimitive
	virtual bool toFile_MeOnly(QFile& out) const override;
    virtual bool fromFile_MeOnly(QFile& in, short dataVersion, int flags, LoadedIDMap& oldToNewIDMap) override;
	virtual bool buildUp() override;

	//! Min corner
	CCVector2 m_minCorner;
	//! Max corner
	CCVector2 m_maxCorner;

	//! Equation coefficients
	PointCoordinateType m_eq[6];

	//! Dimension indexes
	Tuple3ub m_dims;

	//! Min height
	PointCoordinateType m_minZ;
	//! Max height
	PointCoordinateType m_maxZ;
};

#endif // ECV_QUADRIC_PRIMITIVE_HEADER
