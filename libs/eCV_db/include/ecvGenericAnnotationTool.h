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

#ifndef ECV_GENERICANNOTATION_TOOL_HEADER
#define ECV_GENERICANNOTATION_TOOL_HEADER

#include "eCV_db.h"

#include <QFile>

class ccPointCloud;
class ecvGenericVisualizer3D;

//! Generic Annotation Tool interface
class ECV_DB_LIB_API ecvGenericAnnotationTool : public QObject
{
	Q_OBJECT
public:
	//! Default constructor
	/**
		\param mode Annotation mode
	**/
	
	enum AnnotationMode
	{
		SEMANTICS,
		BOUNDINGBOX
	};

	ecvGenericAnnotationTool(AnnotationMode mode = AnnotationMode::BOUNDINGBOX);
	virtual ~ecvGenericAnnotationTool() = default;

	inline AnnotationMode getAnnotationMode() { return m_annotationMode; }

public:
	virtual void setVisualizer(ecvGenericVisualizer3D* viewer = nullptr) = 0;

	virtual bool loadClassesFromFile(const std::string& file) = 0;
	virtual void getAnnotationLabels(std::vector<std::string>& labelList) = 0;
	virtual bool getCurrentAnnotations(std::vector<int>& annos) const = 0;
	virtual void initAnnotationLabels(const std::vector<std::string>& labelList) {}

	virtual void toggleInteractor() = 0;
	virtual bool setInputCloud(ccPointCloud* pointCloud, int viewport = 0) = 0;
	virtual void start() = 0;
	virtual void stop() = 0;

	virtual void intersectMode() = 0;
	virtual void unionMode() = 0;
	virtual void trimMode() = 0;
	virtual void resetMode() = 0;

public:
	virtual void reset() = 0;
	virtual void clear() = 0;
	virtual void exportAnnotations() = 0;
	virtual void updateCloud() = 0;
	virtual void changeAnnotationType(const std::string& type) = 0;
	virtual void selectExistedAnnotation(const std::string& type) = 0;
	virtual void showAnnotation() = 0;
	virtual void hideAnnotation() = 0;
	virtual void showOrigin() = 0;
	virtual void hideOrigin() = 0;
	virtual void removeAnnotation() = 0;

signals:
	void objectPicked(bool isPicked);

protected:
	//! Builds primitive
	/** Transformation will be applied afterwards!
		\return success
	**/
	virtual bool buildUp() { return true; }

    virtual void initialize(ecvGenericVisualizer3D* viewer) = 0;

	//! Returns vertices
	ccPointCloud* vertices();

	AnnotationMode m_annotationMode;
	ccPointCloud* m_associatedCloud;
};

#endif // ECV_GENERICANNOTATION_TOOL_HEADER
