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

#ifndef ECV_2D_VIEWPORT_LABEL_HEADER
#define ECV_2D_VIEWPORT_LABEL_HEADER

//Local
#include "ecv2DViewportObject.h"

//! 2D viewport label
class ECV_DB_LIB_API cc2DViewportLabel : public cc2DViewportObject
{
public:

	//! Default constructor
	explicit cc2DViewportLabel(QString name = QString());

	//inherited from ccHObject
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::VIEWPORT_2D_LABEL; }
	virtual bool isSerializable() const override { return true; }

	//! Returns ROI (relative to screen)
	const float* roi() const { return m_roi; }

	//! Sets ROI (relative to screen)
	void setRoi(const float* roi);

	void clear2Dviews();

	void updateLabel();

	void update2DLabelView(CC_DRAW_CONTEXT& context, bool updateScreen  = true );

protected:

	//inherited from ccHObject
	virtual bool toFile_MeOnly(QFile& out) const override;
    virtual bool fromFile_MeOnly(QFile& in, short dataVersion, int flags, LoadedIDMap& oldToNewIDMap) override;

	//! Draws the entity only (not its children)
	virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;

	//! label ROI
	/** ROI is relative to screen
	**/
	float m_roi[4];
};

#endif // ECV_2D_VIEWPORT_LABEL_HEADER
