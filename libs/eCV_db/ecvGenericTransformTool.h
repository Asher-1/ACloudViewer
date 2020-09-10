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

#ifndef ECV_GENERIC_TRANSFORM_TOOL_HEADER
#define ECV_GENERIC_TRANSFORM_TOOL_HEADER

#include "eCV_db.h"

#include <QFile>
#include <ecvGLMatrix.h>

class ccHObject;
class ecvGenericVisualizer3D;

//! Generic Annotation Tool interface
class ECV_DB_LIB_API ecvGenericTransformTool : public QObject
{
	Q_OBJECT
public:
	//! Default constructor
	/**
		\param mode Annotation mode
	**/
	
	enum RotationMode
	{
		R_XYZ,
		R_X,
		R_Y,
		R_Z
	};

	enum TranslationMOde
	{
		T_X,
		T_Y,
		T_Z,
		T_XY,
		T_XZ,
		T_ZY,
		T_XYZ,
		T_NONE
	};

	ecvGenericTransformTool();
	virtual ~ecvGenericTransformTool() = default;

	ccHObject* getAssociatedEntity() { return m_associatedEntity; }

public:
	virtual void setVisualizer(ecvGenericVisualizer3D* viewer = nullptr) = 0;
	virtual bool setInputData(ccHObject* entity, int viewPort = 0);

	virtual void showInteractor(bool state) = 0;

	virtual bool start() = 0;
	virtual void stop() = 0;
	virtual void reset() = 0;
	virtual void clear() = 0;
	virtual void setTranlationMode(TranslationMOde mode) = 0;
	virtual void setRotationMode(RotationMode mode) = 0;
	virtual void setScaleEnabled(bool state) = 0;
	virtual void setShearEnabled(bool state) = 0;

	virtual const ccGLMatrixd getFinalTransformation() = 0;

	virtual void getOutput(std::vector<ccHObject*>& out) = 0;

signals:
	void tranformMatrix(const ccGLMatrixd& transMatrix);

protected:

	ccHObject* m_associatedEntity;
};

#endif // ECV_GENERIC_TRANSFORM_TOOL_HEADER
