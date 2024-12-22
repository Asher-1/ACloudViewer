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

#include "ecvHObjectCaster.h"

// types
#include "Image.h"
#include "LineSet.h"
#include "Octree.h"
#include "RGBDImage.h"
#include "VoxelGrid.h"
#include "ecv2DLabel.h"
#include "ecv2DViewportLabel.h"
#include "ecv2DViewportObject.h"
#include "ecvBBox.h"
#include "ecvBox.h"
#include "ecvCameraSensor.h"
#include "ecvCone.h"
#include "ecvCoordinateSystem.h"
#include "ecvCylinder.h"
#include "ecvDish.h"
#include "ecvExtru.h"
#include "ecvFacet.h"
#include "ecvGBLSensor.h"
#include "ecvGenericMesh.h"
#include "ecvGenericPointCloud.h"
#include "ecvGenericPrimitive.h"
#include "ecvHObject.h"
#include "ecvImage.h"
#include "ecvIndexedTransformationBuffer.h"
#include "ecvKdTree.h"
#include "ecvMesh.h"
#include "ecvOctree.h"
#include "ecvOctreeProxy.h"
#include "ecvOrientedBBox.h"
#include "ecvPlane.h"
#include "ecvPointCloud.h"
#include "ecvPolyline.h"
#include "ecvQuadric.h"
#include "ecvSensor.h"
#include "ecvShiftedObject.h"
#include "ecvSphere.h"
#include "ecvSubMesh.h"
#include "ecvTorus.h"

/*** helpers ***/

ccPointCloud* ccHObjectCaster::ToPointCloud(
        ccHObject* obj, bool* lockedVertices /*= nullptr*/) {
    if (lockedVertices) {
        *lockedVertices = false;
    }

    if (obj) {
        if (obj->isA(CV_TYPES::POINT_CLOUD)) {
            return static_cast<ccPointCloud*>(obj);
        } else if (obj->isKindOf(CV_TYPES::MESH)) {
            ccGenericPointCloud* vertices =
                    static_cast<ccGenericMesh*>(obj)->getAssociatedCloud();
            if (vertices) {
                if (!obj->isA(CV_TYPES::MESH) &&
                    lockedVertices)  // no need to 'lock' the vertices if the
                                     // user works on the parent mesh
                {
                    *lockedVertices = vertices->isLocked();
                }
                return ccHObjectCaster::ToPointCloud(vertices);
            }
        }
    }

    return nullptr;
}

ccGenericPointCloud* ccHObjectCaster::ToGenericPointCloud(
        ccHObject* obj, bool* lockedVertices /*= nullptr*/) {
    if (lockedVertices) {
        *lockedVertices = false;
    }

    if (obj) {
        if (obj->isKindOf(CV_TYPES::POINT_CLOUD)) {
            return static_cast<ccGenericPointCloud*>(obj);
        } else if (obj->isKindOf(CV_TYPES::MESH)) {
            ccGenericPointCloud* vertices =
                    static_cast<ccGenericMesh*>(obj)->getAssociatedCloud();
            if (vertices) {
                if (!obj->isA(CV_TYPES::MESH) &&
                    lockedVertices)  // no need to 'lock' the vertices if the
                                     // user works on the parent mesh
                {
                    *lockedVertices = vertices->isLocked();
                }
                return vertices;
            }
        }
    }

    return nullptr;
}

ccShiftedObject* ccHObjectCaster::ToShifted(
        ccHObject* obj, bool* lockedVertices /*= nullptr*/) {
    ccGenericPointCloud* cloud = ToGenericPointCloud(obj, lockedVertices);
    if (cloud) return cloud;

    if (obj && obj->isKindOf(CV_TYPES::POLY_LINE)) {
        if (lockedVertices) {
            *lockedVertices = false;
        }
        return static_cast<ccPolyline*>(obj);
    }

    return nullptr;
}

ccGenericMesh* ccHObjectCaster::ToGenericMesh(ccHObject* obj) {
    return (obj && obj->isKindOf(CV_TYPES::MESH)
                    ? static_cast<ccGenericMesh*>(obj)
                    : nullptr);
}

ccMesh* ccHObjectCaster::ToMesh(ccHObject* obj) {
    return (obj && (obj->isA(CV_TYPES::MESH) ||
                    obj->isKindOf(CV_TYPES::PRIMITIVE))
                    ? static_cast<ccMesh*>(obj)
                    : nullptr);
}

ccSubMesh* ccHObjectCaster::ToSubMesh(ccHObject* obj) {
    return (obj && obj->isA(CV_TYPES::SUB_MESH) ? static_cast<ccSubMesh*>(obj)
                                                : nullptr);
}

ccPolyline* ccHObjectCaster::ToPolyline(ccHObject* obj) {
    return (obj && obj->isA(CV_TYPES::POLY_LINE) ? static_cast<ccPolyline*>(obj)
                                                 : nullptr);
}

ccFacet* ccHObjectCaster::ToFacet(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::FACET) ? static_cast<ccFacet*>(obj)
                                            : nullptr;
}

ccPlanarEntityInterface* ccHObjectCaster::ToPlanarEntity(ccHObject* obj) {
    if (obj) {
        if (obj->isA(CV_TYPES::FACET)) {
            return static_cast<ccFacet*>(obj);
        } else if (obj->isA(CV_TYPES::PLANE)) {
            return static_cast<ccPlane*>(obj);
        }
    }
    return nullptr;
}

ccGenericPrimitive* ccHObjectCaster::ToPrimitive(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::PRIMITIVE)
                   ? static_cast<ccGenericPrimitive*>(obj)
                   : nullptr;
}

ccSphere* ccHObjectCaster::ToSphere(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::SPHERE) ? static_cast<ccSphere*>(obj)
                                             : nullptr;
}

ccCylinder* ccHObjectCaster::ToCylinder(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::CYLINDER) ? static_cast<ccCylinder*>(obj)
                                               : nullptr;
}

ccCone* ccHObjectCaster::ToCone(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::CONE) ? static_cast<ccCone*>(obj)
                                                : nullptr;
}

ccQuadric* ccHObjectCaster::ToQuadric(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::QUADRIC)
                   ? static_cast<ccQuadric*>(obj)
                   : nullptr;
}

ccBox* ccHObjectCaster::ToBox(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::BOX) ? static_cast<ccBox*>(obj)
                                               : nullptr;
}

ccPlane* ccHObjectCaster::ToPlane(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::PLANE) ? static_cast<ccPlane*>(obj)
                                            : nullptr;
}

ccDish* ccHObjectCaster::ToDish(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::DISH) ? static_cast<ccDish*>(obj)
                                           : nullptr;
}

ccExtru* ccHObjectCaster::ToExtru(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::EXTRU) ? static_cast<ccExtru*>(obj)
                                            : nullptr;
}

ccTorus* ccHObjectCaster::ToTorus(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::TORUS) ? static_cast<ccTorus*>(obj)
                                            : nullptr;
}

ccOctreeProxy* ccHObjectCaster::ToOctreeProxy(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::POINT_OCTREE)
                   ? static_cast<ccOctreeProxy*>(obj)
                   : nullptr;
}

ccOctree* ccHObjectCaster::ToOctree(ccHObject* obj) {
    ccOctreeProxy* proxy = ToOctreeProxy(obj);
    return proxy ? proxy->getOctree().data() : nullptr;
}

ccKdTree* ccHObjectCaster::ToKdTree(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::POINT_KDTREE) ? static_cast<ccKdTree*>(obj)
                                                   : nullptr;
}

ccSensor* ccHObjectCaster::ToSensor(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::SENSOR) ? static_cast<ccSensor*>(obj)
                                                  : nullptr;
}

ccGBLSensor* ccHObjectCaster::ToGBLSensor(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::GBL_SENSOR)
                   ? static_cast<ccGBLSensor*>(obj)
                   : nullptr;
}

ccCameraSensor* ccHObjectCaster::ToCameraSensor(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::CAMERA_SENSOR)
                   ? static_cast<ccCameraSensor*>(obj)
                   : nullptr;
}

ccImage* ccHObjectCaster::ToImage(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::IMAGE) ? static_cast<ccImage*>(obj)
                                                 : nullptr;
}

cc2DLabel* ccHObjectCaster::To2DLabel(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::LABEL_2D) ? static_cast<cc2DLabel*>(obj)
                                               : nullptr;
}

cc2DViewportLabel* ccHObjectCaster::To2DViewportLabel(ccHObject* obj) {
    return obj && obj->isA(CV_TYPES::VIEWPORT_2D_LABEL)
                   ? static_cast<cc2DViewportLabel*>(obj)
                   : nullptr;
}

cc2DViewportObject* ccHObjectCaster::To2DViewportObject(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::VIEWPORT_2D_OBJECT)
                   ? static_cast<cc2DViewportObject*>(obj)
                   : nullptr;
}

ccIndexedTransformationBuffer* ccHObjectCaster::ToTransBuffer(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::TRANS_BUFFER)
                   ? static_cast<ccIndexedTransformationBuffer*>(obj)
                   : nullptr;
}

using namespace cloudViewer;

geometry::Image* ccHObjectCaster::ToImage2(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::IMAGE2)
                   ? static_cast<geometry::Image*>(obj)
                   : nullptr;
}

geometry::RGBDImage* ccHObjectCaster::ToRGBDImage(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::RGBD_IMAGE)
                   ? static_cast<geometry::RGBDImage*>(obj)
                   : nullptr;
}

geometry::VoxelGrid* ccHObjectCaster::ToVoxelGrid(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::VOXEL_GRID)
                   ? static_cast<geometry::VoxelGrid*>(obj)
                   : nullptr;
}

geometry::LineSet* ccHObjectCaster::ToLineSet(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::LINESET)
                   ? static_cast<geometry::LineSet*>(obj)
                   : nullptr;
}

geometry::Octree* ccHObjectCaster::ToOctree2(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::POINT_OCTREE2)
                   ? static_cast<geometry::Octree*>(obj)
                   : nullptr;
}

ccBBox* ccHObjectCaster::ToBBox(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::BBOX) ? static_cast<ccBBox*>(obj)
                                                : nullptr;
}

ecvOrientedBBox* ccHObjectCaster::ToOrientedBBox(ccHObject* obj) {
    return obj && obj->isKindOf(CV_TYPES::ORIENTED_BBOX)
                   ? static_cast<ecvOrientedBBox*>(obj)
                   : nullptr;
}

ccCoordinateSystem* ccHObjectCaster::ToCoordinateSystem(ccHObject* obj) {
    return (obj && obj->isKindOf(CV_TYPES::COORDINATESYSTEM)
                    ? static_cast<ccCoordinateSystem*>(obj)
                    : nullptr);
}

bool ccHObjectCaster::CloneChildren(
        const ccHObject* sourceEntity,
        ccHObject* destEntity,
        std::vector<int>* newPointOrTriangleIndex /*=nullptr*/,
        const ccHObject* sourceEntityProxy /*=nullptr*/,
        ccHObject* destEntityProxy /*=nullptr*/) {
    if (!sourceEntity || !destEntity) {
        assert(false);
        return false;
    }

    bool sourceIsCloud = sourceEntity->isKindOf(CV_TYPES::POINT_CLOUD);
    bool destIsCloud = destEntity->isKindOf(CV_TYPES::POINT_CLOUD);
    bool sourceAndDestAreCloud = sourceIsCloud && destIsCloud;

    bool sourceIsMesh = sourceEntity->isKindOf(CV_TYPES::MESH);
    bool destIsMesh = destEntity->isKindOf(CV_TYPES::MESH);
    bool sourceAndDestAreMeshes = sourceIsMesh && destIsMesh;

    unsigned numberOfPointOrTriangle = 0;
    if (sourceIsCloud)
        numberOfPointOrTriangle =
                static_cast<const ccGenericPointCloud*>(sourceEntity)->size();
    else if (sourceIsMesh)
        numberOfPointOrTriangle =
                static_cast<const ccGenericMesh*>(sourceEntity)->size();

    if (newPointOrTriangleIndex) {
        if (sourceEntity == destEntity) {
            CVLog::Warning(
                    "[ccHObjectCaster::CloneChildren] Providing a "
                    "point/triangle correspondance map while the source and "
                    "destination entities are the same");
            // we can live with that...
        }

        if (!sourceAndDestAreCloud && !sourceAndDestAreMeshes) {
            CVLog::Warning(
                    "[ccHObjectCaster::CloneChildren] A point/triangle "
                    "correspondance map can only work between 2 entities of "
                    "the same type");
            return false;
        }

        if (newPointOrTriangleIndex->size() != numberOfPointOrTriangle) {
            CVLog::Warning(
                    "[ccHObjectCaster::CloneChildren] Mismatch between the "
                    "point/triangle correspondance map and the source entity "
                    "size");
            return false;
        }
    }

    QMap<ccCameraSensor*, ccCameraSensor*> clonedCameraSensors;

    const ccHObject* currentSourceEntity =
            (sourceEntityProxy ? sourceEntityProxy : sourceEntity);
    ccHObject* currentDestEntity =
            (destEntityProxy ? destEntityProxy : destEntity);

    // for each child
    for (unsigned i = 0; i < currentSourceEntity->getChildrenNumber(); ++i) {
        ccHObject* child = currentSourceEntity->getChild(i);

        switch (child->getClassID()) {
            // 2D Label
            case CV_TYPES::LABEL_2D: {
                cc2DLabel* label = static_cast<cc2DLabel*>(child);

                // check if we can keep this label
                bool keepThisLabel = true;
                if (newPointOrTriangleIndex) {
                    for (unsigned i = 0; i < label->size(); ++i) {
                        const cc2DLabel::PickedPoint& pp =
                                label->getPickedPoint(i);
                        if (pp.entity() == sourceEntity &&
                            newPointOrTriangleIndex->at(pp.index) < 0) {
                            // this label relies on a point or triangle that has
                            // no correspondance in the destination entity
                            keepThisLabel = false;
                            break;
                        }
                    }
                }

                if (keepThisLabel) {
                    cc2DLabel* clonedLabel = new cc2DLabel(*label);

                    for (unsigned i = 0; i < label->size(); ++i) {
                        cc2DLabel::PickedPoint pp = label->getPickedPoint(i);
                        if (pp.entity() == sourceEntity) {
                            if (sourceIsCloud)
                                pp.cloud = static_cast<ccGenericPointCloud*>(
                                        destEntity);
                            else
                                pp.mesh =
                                        static_cast<ccGenericMesh*>(destEntity);

                            if (newPointOrTriangleIndex) {
                                pp.index = static_cast<unsigned>(
                                        newPointOrTriangleIndex->at(
                                                pp.index));  // we've checked
                                                             // above that it's
                                                             // >= 0
                            }
                        }
                        clonedLabel->addPickedPoint(pp);
                    }
                    clonedLabel->setName(
                            label->getName());  // the label name is overridden
                                                // by calls to addPickedPoint

                    currentDestEntity->addChild(clonedLabel);
                }
            } break;

            // Image
            case CV_TYPES::IMAGE: {
                ccImage* image = static_cast<ccImage*>(child);
                ccImage* clonedImage = new ccImage(*image);

                ccCameraSensor* camSensor = image->getAssociatedSensor();
                if (camSensor) {
                    if (clonedCameraSensors.contains(camSensor)) {
                        // if we have already cloned the sensor on which this
                        // image depends, we can simply update the link
                        clonedImage->setAssociatedSensor(
                                clonedCameraSensors[camSensor]);
                    } else {
                        // else we have to clone the sensor
                        ccCameraSensor* clonedCamSensor =
                                new ccCameraSensor(*camSensor);
                        clonedCameraSensors.insert(camSensor, clonedCamSensor);

                        clonedImage->setAssociatedSensor(clonedCamSensor);
                        clonedImage->addChild(clonedCamSensor);
                    }
                }

                currentDestEntity->addChild(clonedImage);
            } break;

            // Camera sensor
            case CV_TYPES::CAMERA_SENSOR: {
                ccCameraSensor* camSensor = static_cast<ccCameraSensor*>(child);
                ccCameraSensor* clonedCamSensor =
                        new ccCameraSensor(*camSensor);
                clonedCameraSensors.insert(camSensor, clonedCamSensor);

                currentDestEntity->addChild(clonedCamSensor);
            } break;

            // GBL sensor
            case CV_TYPES::GBL_SENSOR: {
                ccGBLSensor* gblSensor = static_cast<ccGBLSensor*>(child);
                ccGBLSensor* clonedGBLSensor =
                        new ccGBLSensor(*gblSensor);

                currentDestEntity->addChild(clonedGBLSensor);
            } break;

            // 2D Viewport object
            case CV_TYPES::VIEWPORT_2D_OBJECT: {
                cc2DViewportObject* viewportObject =
                        static_cast<cc2DViewportObject*>(child);
                ;
                cc2DViewportObject* clonedViewportObject =
                        new cc2DViewportObject(*viewportObject);

                currentDestEntity->addChild(clonedViewportObject);
            } break;

            // 2D Viewport label
            case CV_TYPES::VIEWPORT_2D_LABEL: {
                cc2DViewportLabel* viewportLabel =
                        static_cast<cc2DViewportLabel*>(child);
                ;
                cc2DViewportLabel* clonedViewportLabel =
                        new cc2DViewportLabel(*viewportLabel);

                currentDestEntity->addChild(clonedViewportLabel);
            } break;

            // Groups
            case CV_TYPES::HIERARCHY_OBJECT: {
                ccHObject* newGroup = new ccHObject(*child);
                // start (or proceed with) the recursion
                if (CloneChildren(sourceEntity, destEntity,
                                  newPointOrTriangleIndex, child, newGroup)) {
                    if (newGroup->getChildrenNumber() != 0) {
                        currentDestEntity->addChild(newGroup);
                    } else {
                        // empty group, no need to keep it
                        delete newGroup;
                        newGroup = nullptr;
                    }
                } else {
                    // something bad happened
                    return false;
                }
            } break;

            // Meshes
            case CV_TYPES::MESH: {
                // TODO
            } break;

            default: {
                // nothing to do
            } break;
        }
    }

    return true;
}
