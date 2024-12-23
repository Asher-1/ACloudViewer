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

#ifndef ECV_INDEXED_TRANSFORMATION_BUFFER_HEADER
#define ECV_INDEXED_TRANSFORMATION_BUFFER_HEADER

//Local
#include "ecvIndexedTransformation.h"
#include "ecvHObject.h"
#include "ecvBBox.h"

//system
#include <float.h>

//! Indexed Transformation buffer
class ECV_DB_LIB_API ccIndexedTransformationBuffer : public ccHObject, public std::vector<ccIndexedTransformation>
{
public:

	//! Default constructor
	ccIndexedTransformationBuffer(QString name = QString("Trans. buffer"));
	//! Copy constructor
	ccIndexedTransformationBuffer(const ccIndexedTransformationBuffer& buffer);

	//inherited from ccHObject
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::TRANS_BUFFER; }
	virtual bool isSerializable() const override { return true; }

	//! Sorts transformations based on their index
	/** Ascending sort.
	**/
	void sort();

	//! Returns the nearest indexed transformation(s) to a given index
	/** This method returns the preceding and following transformations.
		
		\warning Binary search: buffer must be sorted! (see ccIndexedTransformationBuffer::sort)
		
		\param index query index (e.g. timestamp)
		\param trans1 directly preceding transformation (if any - null otherwise)
		\param trans2 directly following transformation (if any - null otherwise)
		\param trans1IndexInBuffer (optional) index of trans1 in buffer
		\param trans2IndexInBuffer (optional) index of trans2 in buffer
		\return success
	**/
	bool findNearest(double index,
		const ccIndexedTransformation* &trans1,
		const ccIndexedTransformation* &trans2,
		size_t* trans1IndexInBuffer = 0,
		size_t* trans2IndexInBuffer = 0) const;

	//! Returns the indexed transformation at a given index (interpolates it if necessary)
	/** \warning Binary search: buffer must be sorted! (see ccIndexedTransformationBuffer::sort)

		\param index query index (e.g. timestamp)
		\param trans output transformation (if successful)
		\param maxIndexDistForInterpolation max 'distance' between query index and existing indexes to actually interpolate/output a transformation
		\return success
	**/
	bool getInterpolatedTransformation(	double index,
										ccIndexedTransformation& trans,
										double maxIndexDistForInterpolation = DBL_MAX) const;

	//! [Display option] Returns whether trihedrons should be displayed or not (otherwise only points or a polyline)
	bool triherdonsShown() { return m_showTrihedrons; }
	//! [Display option] Sets whether trihedrons should be displayed or not (otherwise only points or a polyline)
	void showTriherdons(bool state) { m_showTrihedrons = state; }

	//! [Display option] Returns trihedron display size
	float triherdonsDisplayScale() { return m_trihedronsScale; }
	//! [Display option] Sets trihedron display size
	void setTriherdonsDisplayScale(float scale) { m_trihedronsScale = scale; }

	//! [Display option] Returns whether the path should be displayed as a polyline or not (otherwise only points)
	bool isPathShonwAsPolyline() { return m_showAsPolyline; }
	//! [Display option] Sets whether the path should be displayed as a polyline or not (otherwise only points)
	void showPathAsPolyline(bool state) { m_showAsPolyline = state; }

	//! Invalidates the bounding box
	/** Should be called whenever the content of this structure changes!
	**/
	void invalidateBoundingBox();

	//Inherited from ccHObject
	virtual ccBBox getOwnBB(bool withGLFeatures = false) override;

protected:

	//inherited from ccHObject
	virtual bool toFile_MeOnly(QFile& out) const override;
    virtual bool fromFile_MeOnly(QFile& in, short dataVersion, int flags, LoadedIDMap& oldToNewIDMap) override;

	//! Bounding box
	ccBBox m_bBox;

	//! Bounding box last 'validity' size
	size_t m_bBoxValidSize;

	//! Whether the path should be displayed as a polyline or not
	bool m_showAsPolyline;
	//! Whether trihedrons should be displayed or not
	bool m_showTrihedrons;
	//! Trihedrons display scale
	float m_trihedronsScale;
};

#endif // ECV_INDEXED_TRANSFORMATION_BUFFER_HEADER
