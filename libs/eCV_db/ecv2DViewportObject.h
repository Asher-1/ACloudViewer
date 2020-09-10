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

#ifndef ECV_2D_VIEWPORT_OBJECT_HEADER
#define ECV_2D_VIEWPORT_OBJECT_HEADER

//Local
#include "ecvHObject.h"
#include "ecvViewportParameters.h"

//! 2D viewport object
class ECV_DB_LIB_API cc2DViewportObject : public ccHObject
{
public:

	//! Default constructor
	cc2DViewportObject(QString name = QString());

	//inherited from ccHObject
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::VIEWPORT_2D_OBJECT; }
	virtual bool isSerializable() const override { return true; }

	//! Sets perspective view state
	void setParameters(const ecvViewportParameters& params) { m_params = params; }

	//! Gets parameters
	const ecvViewportParameters& getParameters() const { return m_params; }

protected:

	//inherited from ccHObject
	virtual bool toFile_MeOnly(QFile& out) const override;
	virtual bool fromFile_MeOnly(QFile& in, short dataVersion, int flags) override;

	//! Viewport parameters
	ecvViewportParameters m_params;
};

#endif //CC_2D_VIEWPORT_OBJECT_HEADER
