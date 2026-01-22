// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccItemTree.h"

#include "ccMPlaneErrors.h"

// CV_DB_LIB
#include <ecvDisplayTools.h>

bool compareObjects(QString name,
                    ccHObject *object,
                    CC_ITEM_TREE::CC_TYPES_MAP type) {
    if (!object) {
        return false;
    }
    if (object->isKindOf(type) &&
        (name == nullptr || name != nullptr && (object->getName() == name))) {
        return true;
    }
    return false;
}

ccHObject *CC_ITEM_TREE::findItemInHierarchy(ccHObject *root,
                                             CC_TYPES_MAP itemType,
                                             QString itemName) {
    if (!root) {
        throw MplaneInvalidArgument("root is not allowed to be nullptr");
    }
    auto childrenNumber = root->getChildrenNumber();
    for (unsigned int i = 0; i < childrenNumber; ++i) {
        ccHObject *child = root->getChild(i);
        if (compareObjects(itemName, child, itemType)) {
            return child;
        }
        if (child->getChildrenNumber() > 0) {
            ccHObject *subChild =
                    findItemInHierarchy(child, itemType, itemName);
            if (compareObjects(itemName, subChild, itemType)) {
                return subChild;
            }
        }
    }
    return nullptr;
}

ccHObject *CC_ITEM_TREE::findOrCreateContainer(ccHObject *parent,
                                               const QString &containerName,
                                               ecvMainAppInterface *app) {
    if (!parent) {
        throw MplaneInvalidArgument("parent is not allowed to be nullptr");
    }
    if (!app) {
        throw MplaneInvalidArgument("app is not allowed to be nullptr");
    }
    if (containerName.isEmpty()) {
        throw MplaneInvalidArgument("containerName is not allowed to be empty");
    }
    ccHObject *container = CC_ITEM_TREE::findItemInHierarchy(
            parent, CC_ITEM_TREE::CC_TYPES_MAP::HIERARCHY_OBJECT,
            containerName);
    if (container) {
        return container;
    }
    container = new ccHObject(containerName);
    parent->addChild(container);
    app->addToDB(container, false, true, false, false);

    return container;
}

cc2DLabel *CC_ITEM_TREE::createPointLabel2D(QString labelText,
                                            ccPointCloud *sourceCloud,
                                            QPoint clickPoint,
                                            unsigned int pointIdx) {
    if (labelText.isEmpty()) {
        throw MplaneInvalidArgument("labelText is not allowed to be empty");
    }
    if (!sourceCloud) {
        throw MplaneInvalidArgument("sourceCloud is not allowed to be nullptr");
    }
    cc2DLabel *newLabel = nullptr;
    // ccGenericGLDisplay* display = sourceCloud->getDisplay();
    if (ecvDisplayTools::GetCurrentScreen()) {
        newLabel = new cc2DLabel();
        newLabel->addPickedPoint(sourceCloud, pointIdx);
        newLabel->setName(labelText);
        newLabel->setVisible(true);
        newLabel->setDisplayedIn2D(false);
        newLabel->displayPointLegend(true);
        newLabel->setCollapsed(true);
        // newLabel->setDisplay(display);

        QSize size = ecvDisplayTools::GetScreenSize();
        newLabel->setPosition(
                static_cast<float>(clickPoint.x() + 20) / size.width(),
                static_cast<float>(clickPoint.y() + 20) / size.height());
        newLabel->redrawDisplay();
    }
    return newLabel;
}
