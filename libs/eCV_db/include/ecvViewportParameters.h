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

#ifndef ECV_VIEWPORT_PARAMETERS_HEADER
#define ECV_VIEWPORT_PARAMETERS_HEADER

//Local
#include "eCV_db.h"
#include "ecvGLMatrix.h"
#include "ecvSerializableObject.h"

//! Standard parameters for GL displays/viewports
class ECV_DB_LIB_API ecvViewportParameters : public ccSerializableObject
{
public:
	//! Default constructor
	ecvViewportParameters();

	//! Copy constructor
	ecvViewportParameters(const ecvViewportParameters& params);

	//inherited from ccSerializableObject
	bool isSerializable() const override { return true; }
	bool toFile(QFile& out) const override;
	bool fromFile(QFile& in, short dataVersion, int flags) override;

	//! Current pixel size (in 'current unit'/pixel)
	/** This scale is valid eveywhere in ortho. mode
		or at the focal distance in perspective mode.
		Warning: doesn't take current zoom into account!
	**/
	float pixelSize;

	//! Current zoom
	float zoom;

	//! Visualization matrix (rotation only)
	ccGLMatrixd viewMat;

	//! Point size
	float defaultPointSize;
	//! Line width
	float defaultLineWidth;

	//! Perspective view state
	bool perspectiveView;
	//! Whether view is centered on displayed scene (true) or on the user eye (false)
	/** Always true for ortho. mode.
	**/
	bool objectCenteredView;

	//! Theoretical perspective 'zNear' relative position
	double zNearCoef;
	//! Actual perspective 'zNear' value
	double zNear;
	//! Actual perspective 'zFar' value
	double zFar;

	//! Rotation pivot point (for object-centered view modes)
	CCVector3d pivotPoint;

	//! Camera center (for perspective mode)
	CCVector3d cameraCenter;

	CCVector3d focal;

	CCVector3d up;

	//! Camera F.O.V. (field of view - for perspective mode only)
	float fov;
	//! Camera aspect ratio (perspective mode only)
	float perspectiveAspectRatio;

	//! 3D view aspect ratio (ortho mode only)
	/** AR = width / height
	**/
	float orthoAspectRatio;

	//! Helper: converts an integer (increment) in [0 iMax] to a double (zNear) value in [0.001 1]
	static double IncrementToZNearCoef(int i, int iMax)
	{
		assert(i >= 0 && i <= iMax);
		return pow(10, -static_cast<double>((iMax - i) * 3) / iMax); //between 10^-3 and 1
	}

	//! Helper: converts a double (zNear) value in [0 1] to integer increments in [0 iMax]
	static int ZNearCoefToIncrement(double coef, int iMax)
	{
		assert(coef >= 0 && coef <= 1.0);
		double id = -(iMax / 3.0) * log10(coef);
		int i = static_cast<int>(id);
		//cope with numerical inaccuracies
		if (fabs(id - i) > fabs(id - (i + 1)))
		{
			++i;
		}
		assert(i >= 0 && i <= iMax);
		return iMax - i;
	}
};

#endif // ECV_VIEWPORT_PARAMETERS_HEADER
