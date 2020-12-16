//##########################################################################
//#                                                                        #
//#                    CLOUDVIEWER PLUGIN: qMPlane                         #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#         COPYRIGHT: AIT Austrian Institute of Technology GmbH           #
//#                                                                        #
//##########################################################################

#ifndef CC_MPLANE_ITEM_TREE
#define CC_MPLANE_ITEM_TREE

// std
#include <array>

// Qt
#include "ecvHObject.h"
#include "ecvMainAppInterface.h"
#include "ecv2DLabel.h"
#include "ecvPointCloud.h"

namespace CC_ITEM_TREE {
	enum CC_TYPES_MAP {
		HIERARCHY_OBJECT = CV_TYPES::HIERARCHY_OBJECT,
		PLANE = CV_TYPES::PLANE,
		LABEL2D = CV_TYPES::LABEL_2D
	};

	/**
	* Find an item in the root's hierarchy tree by type and optionally by name.
	* If multiple matching objects exist in the hierarchy, the first one is returned.
	*
	* @param[in] root The root of the item tree to be searched.
	* @param[in] itemType The type of items to be found.
	* @param[in] itemName (Optional) name of the searched item.
	* @throws std::invalid_argument Exception is thrown if root is nullptr
	* @return ccHObject* Returns an item if found, otherwise a nullptr.
	*/
	ccHObject* findItemInHierarchy(ccHObject *root, CC_TYPES_MAP itemType, QString itemName = nullptr);

	/**
	* Find a matching container or create a new one if it is not existing in the parent's hierarchy tree.
	* The newly created object is added to the parent object as a child.
	*
	* @param[in] parent The parent object of the container.
	* @param[in] containerName The name of the new container object.
	* @param[in] app The ecvMainAppInterface object.
	* @throws std::invalid_argument Exception is thrown if parent is nullptr
	* @return ccHObject* Returns the newly created container if not existing, otherwise existing one.
	*/

	ccHObject* findOrCreateContainer(ccHObject *parent, const QString &containerName, ecvMainAppInterface *app);

	/**
	* Create a cc2DLabel associated with a QPoint within a ccPointCloud.
	* @param[in] labelText The label text to be displayed.
	* @param[in] parentCloud The associated parent cloud.
	* @param[in] clickPoint The 2D clickPoint which specifies the rendered label positions.
	* @param[in] pointIdx The point index which specifies the 3D point within the parent cloud.
	*/
	cc2DLabel* createPointLabel2D(QString labelText, ccPointCloud *parentCloud, QPoint clickPoint, unsigned int pointIdx);
}

#endif
