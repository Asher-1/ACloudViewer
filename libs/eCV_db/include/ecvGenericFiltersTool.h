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

#ifndef ECV_GENERIC_FILTERS_TOOL_HEADER
#define ECV_GENERIC_FILTERS_TOOL_HEADER

#include "eCV_db.h"

#include "ecvBBox.h"

// QT
#include <QFile>
#include <QObject>

class ccHObject;
class ccPolyline;
class ecvGenericVisualizer3D;

//! Generic Filters Tool interface
class ECV_DB_LIB_API ecvGenericFiltersTool : public QObject
{
	Q_OBJECT
public:
	enum FilterType
	{
		CLIP_FILTER,
		SLICE_FILTER,
		DECIMATE_FILTER,
		ISOSURFACE_FILTER,
		THRESHOLD_FILTER,
		SMOOTH_FILTER,
		PROBE_FILTER,	
		STREAMLINE_FILTER,
		GLYPH_FILTER
	};

	//! Default constructor
	/**
		\param mode FilterType mode
	**/
	ecvGenericFiltersTool(FilterType mode = FilterType::CLIP_FILTER);

	//! Desctructor
	virtual ~ecvGenericFiltersTool() = default;

	virtual void showInteractor(bool state) = 0;
	virtual bool setInputData(ccHObject* entity, int viewPort = 0) = 0;
	virtual void unregisterFilter() = 0;

	virtual void intersectMode() = 0;
	virtual void unionMode() = 0;
	virtual void trimMode() = 0;
	virtual void resetMode() = 0;

public:
	virtual bool start() = 0;
	virtual void reset() = 0;
	virtual void restore() = 0;

	virtual void clear() = 0;
	virtual void update();
	virtual void getOutput(
		std::vector<ccHObject*>& outputSlices,
		std::vector<ccPolyline*>& outputContours) const = 0;
	virtual ccHObject* getOutput() const = 0;

	virtual void setNegative(bool state) = 0;

	//! Whether to show the box or not
	virtual void showOutline(bool state) = 0;

	virtual QWidget* getFilterWidget() = 0;

	//! Returns the box extents
	virtual const ccBBox& getBox() = 0;

	//! Sets the box extents
	virtual void setBox(const ccBBox& box) = 0;

	//! Shifts the current box
	virtual void shift(const CCVector3& v) = 0;

	//! Manually sets the box parameters
	virtual void set(const ccBBox& extents, const ccGLMatrix& transformation) = 0;
	virtual void get(ccBBox& extents, ccGLMatrix& transformation) = 0;

signals:
	//! Signal sent each time the box is modified
	void boxModified(const ccBBox* box);

public:
	inline FilterType getFilterType() { return m_filterType; }

protected:
	//! Builds primitive
	/** Transformation will be applied afterwards!
		\return success
	**/
	virtual bool buildUp() { return true; }

    virtual void initialize(ecvGenericVisualizer3D* viewer = nullptr) = 0;

	bool m_showOutline = false;

	ccBBox m_box;

	FilterType m_filterType;
	ccHObject* m_associatedEntity;
};

#endif // ECV_GENERIC_FILTERS_TOOL_HEADER
