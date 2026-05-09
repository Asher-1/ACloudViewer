// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvHObject.h"

#include "ecvBBox.h"

// Objects handled by factory
#include <ecvDisplayTypes.h>
#include <ecvGenericDisplayTools.h>
#include <ecvGenericGLDisplay.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

#include "ecv2DLabel.h"
#include "ecv2DViewportLabel.h"
#include "ecvBox.h"
#include "ecvCameraSensor.h"
#include "ecvCircle.h"
#include "ecvCoordinateSystem.h"
#include "ecvCustomObject.h"
#include "ecvCylinder.h"
#include "ecvDisc.h"
#include "ecvDish.h"
#include "ecvExternalFactory.h"
#include "ecvExtru.h"
#include "ecvFacet.h"
#include "ecvGBLSensor.h"
#include "ecvGuiParameters.h"
#include "ecvHObjectCaster.h"
#include "ecvImage.h"
#include "ecvIndexedTransformationBuffer.h"
#include "ecvKdTree.h"
#include "ecvOctree.h"
#include "ecvOctreeProxy.h"
#include "ecvMaterialSet.h"
#include "ecvMeshGroup.h"
#include "ecvPlane.h"
#include "ecvPointCloud.h"
#include "ecvPolyline.h"
#include "ecvQuadric.h"
#include "ecvRepresentationManager.h"
#include "ecvSphere.h"
#include "ecvSubMesh.h"
#include "ecvTorus.h"
#include "ecvViewRepresentation.h"

// CV_CORE_LIB
#include <CVTools.h>
#include <Eigen.h>
#include <Logging.h>

#include <QApplication>

namespace {
ENTITY_TYPE convertClassToEntityType(CV_CLASS_ENUM type) {
    switch (type) {
        case CV_TYPES::HIERARCHY_OBJECT:
            return ENTITY_TYPE::ECV_HIERARCHY_OBJECT;
        case CV_TYPES::POINT_CLOUD:
            return ENTITY_TYPE::ECV_POINT_CLOUD;
        case CV_TYPES::POLY_LINE:
        case CV_TYPES::LINESET:
        case (CV_TYPES::CUSTOM_H_OBJECT | CV_TYPES::POLY_LINE):
            return ENTITY_TYPE::ECV_SHAPE;
        case CV_TYPES::LABEL_2D:
            return ENTITY_TYPE::ECV_2DLABLE;
        case CV_TYPES::VIEWPORT_2D_LABEL:
            return ENTITY_TYPE::ECV_2DLABLE_VIEWPORT;
        case CV_TYPES::POINT_OCTREE:
            return ENTITY_TYPE::ECV_OCTREE;
        case CV_TYPES::POINT_KDTREE:
            return ENTITY_TYPE::ECV_KDTREE;
        case CV_TYPES::FACET:
        case CV_TYPES::PRIMITIVE:
        case CV_TYPES::MESH:
        case CV_TYPES::SUB_MESH:
        case CV_TYPES::SPHERE:
        case CV_TYPES::CONE:
        case CV_TYPES::PLANE:
        case CV_TYPES::CYLINDER:
        case CV_TYPES::TORUS:
        case CV_TYPES::EXTRU:
        case CV_TYPES::DISH:
        case CV_TYPES::DISC:
        case CV_TYPES::BOX:
        case CV_TYPES::COORDINATESYSTEM:
        case CV_TYPES::QUADRIC:
            return ENTITY_TYPE::ECV_MESH;
        case CV_TYPES::IMAGE:
            return ENTITY_TYPE::ECV_IMAGE;
        case CV_TYPES::SENSOR:
        case CV_TYPES::GBL_SENSOR:
        case CV_TYPES::CAMERA_SENSOR:
            return ENTITY_TYPE::ECV_SENSOR;
        default:
            return ENTITY_TYPE::ECV_NONE;
    }
}
}  // namespace

#include <Eigen/Dense>
#include <numeric>

// Qt
#include <QIcon>

namespace {

/// Prefer the draw context display, otherwise the object's view, otherwise
/// ecvViewManager::getEffectiveView().
ecvGenericGLDisplay* mergeDisplay(ecvGenericGLDisplay* ctxDisp,
                                  const ccHObject* obj) {
    if (ctxDisp) return ctxDisp;
    ecvGenericGLDisplay* v =
            obj ? const_cast<ecvGenericGLDisplay*>(obj->getDisplay()) : nullptr;
    if (!v) {
        v = ecvViewManager::instance().getEffectiveView();
    }
    return v;
}

}  // namespace

ccHObject::ccHObject(QString name /*=QString()*/)
    : ccObject(name),
      ccDrawableObject(),
      m_parent(nullptr),
      m_selectionBehavior(SELECTION_AA_BBOX),
      m_isDeleting(false) {
    setVisible(false);
    lockVisibility(true);
    m_glTransHistory.toIdentity();
}

ccHObject::ccHObject(const ccHObject& object)
    : ccObject(object),
      ccDrawableObject(object),
      m_parent(nullptr),
      m_selectionBehavior(object.m_selectionBehavior),
      m_isDeleting(false) {
    m_glTransHistory.toIdentity();
}

ccHObject::~ccHObject() {
    m_isDeleting = true;

    // Clean up per-view representations so that the representation
    // manager does not hold dangling entity pointers after deletion.
    ecvRepresentationManager::instance().removeRepresentationsForEntity(this);

    // process dependencies
    for (std::map<ccHObject*, int>::const_iterator it = m_dependencies.begin();
         it != m_dependencies.end(); ++it) {
        assert(it->first);
        // notify deletion to other object?
        if ((it->second & DP_NOTIFY_OTHER_ON_DELETE) ==
            DP_NOTIFY_OTHER_ON_DELETE) {
            it->first->onDeletionOf(this);
        }

        // delete other object?
        if ((it->second & DP_DELETE_OTHER) == DP_DELETE_OTHER) {
            it->first->removeDependencyFlag(
                    this,
                    DP_NOTIFY_OTHER_ON_DELETE);  // in order to avoid any loop!
            // delete object
            if (it->first->isShareable())
                dynamic_cast<CCShareable*>(it->first)->release();
            else
                delete it->first;
        }
    }
    m_dependencies.clear();

    removeAllChildren();
}

void ccHObject::notifyGeometryUpdate() {
    ecvGenericGLDisplay* disp = getDisplay();
    if (!disp) disp = ecvViewManager::instance().getEffectiveView();
    if (disp) {
        disp->invalidateViewport();
        disp->deprecate3DLayer();
        CC_DRAW_CONTEXT bbCtx;
        bbCtx.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
        bbCtx.removeViewID = QString("BBox-") + getViewId();
        bbCtx.display = disp;
        if (bbCtx.display) bbCtx.display->removeEntities(bbCtx);
    }

    // process dependencies
    for (std::map<ccHObject*, int>::const_iterator it = m_dependencies.begin();
         it != m_dependencies.end(); ++it) {
        assert(it->first);
        // notify deletion to other object?
        if ((it->second & DP_NOTIFY_OTHER_ON_UPDATE) ==
            DP_NOTIFY_OTHER_ON_UPDATE) {
            it->first->onUpdateOf(this);
        }
    }
}

ccHObject* ccHObject::New(CV_CLASS_ENUM objectType, const char* name /*=0*/) {
    switch (objectType) {
        case CV_TYPES::HIERARCHY_OBJECT:
            return new ccHObject(name);
        case CV_TYPES::POINT_CLOUD:
            return new ccPointCloud(name);
        case CV_TYPES::MESH:
            // warning: no associated vertices --> retrieved later
            return new ccMesh(nullptr);
        case CV_TYPES::SUB_MESH:
            // warning: no associated mesh --> retrieved later
            return new ccSubMesh(nullptr);
        case CV_TYPES::MESH_GROUP:
            // warning: deprecated
            CVLog::Warning("[ccHObject::New] Mesh groups are deprecated!");
            // warning: no associated vertices --> retrieved later
            return new ccMeshGroup();
        case CV_TYPES::POLY_LINE:
            // warning: no associated vertices --> retrieved later
            return new ccPolyline(nullptr);
        case CV_TYPES::CIRCLE:
            return new ccCircle();
        case CV_TYPES::FACET:
            return new ccFacet();
        case CV_TYPES::MATERIAL_SET:
            return new ccMaterialSet();
        case CV_TYPES::NORMALS_ARRAY:
            return new NormsTableType();
        case CV_TYPES::NORMAL_INDEXES_ARRAY:
            return new NormsIndexesTableType();
        case CV_TYPES::RGB_COLOR_ARRAY:
            return new ColorsTableType();
        case CV_TYPES::TEX_COORDS_ARRAY:
            return new TextureCoordsContainer();
        case CV_TYPES::IMAGE:
            return new ccImage();
        case CV_TYPES::CALIBRATED_IMAGE:
            return nullptr;  // deprecated
        case CV_TYPES::GBL_SENSOR:
            // warning: default sensor type set in constructor (see
            // CCLib::GroundBasedLidarSensor::setRotationOrder)
            return new ccGBLSensor();
        case CV_TYPES::CAMERA_SENSOR:
            return new ccCameraSensor();
        case CV_TYPES::LABEL_2D:
            return new cc2DLabel(name);
        case CV_TYPES::VIEWPORT_2D_OBJECT:
            return new cc2DViewportObject(name);
        case CV_TYPES::VIEWPORT_2D_LABEL:
            return new cc2DViewportLabel(name);
        case CV_TYPES::PLANE:
            return new ccPlane(name);
        case CV_TYPES::SPHERE:
            return new ccSphere(name);
        case CV_TYPES::TORUS:
            return new ccTorus(name);
        case CV_TYPES::CYLINDER:
        case CV_TYPES::OLD_CYLINDER_ID:
            return new ccCylinder(name);
        case CV_TYPES::BOX:
            return new ccBox(name);
        case CV_TYPES::CONE:
            return new ccCone(name);
        case CV_TYPES::DISC:
            return new ccDisc(name);
        case CV_TYPES::DISH:
            return new ccDish(name);
        case CV_TYPES::EXTRU:
            return new ccExtru(name);
        case CV_TYPES::QUADRIC:
            return new ccQuadric(name);
        case CV_TYPES::TRANS_BUFFER:
            return new ccIndexedTransformationBuffer(name);
        case CV_TYPES::CUSTOM_H_OBJECT:
            return new ccCustomHObject(name);
        case CV_TYPES::CUSTOM_LEAF_OBJECT:
            return new ccCustomLeafObject(name);
        case CV_TYPES::COORDINATESYSTEM:
            return new ccCoordinateSystem(name);
        case CV_TYPES::POINT_OCTREE:
        case CV_TYPES::POINT_KDTREE:
            // construction this way is not supported (yet)
            CVLog::ErrorDebug(
                    "[ccHObject::New] This object (type %i) can't be "
                    "constructed this way (yet)!",
                    objectType);
            break;
        default:
            // unhandled ID
            CVLog::ErrorDebug("[ccHObject::New] Invalid object type (%i)!",
                              objectType);
            break;
    }

    return nullptr;
}

ccHObject* ccHObject::New(const QString& pluginId,
                          const QString& classId,
                          const char* name) {
    ccExternalFactory::Container::Shared externalFactories =
            ccExternalFactory::Container::GetUniqueInstance();
    if (!externalFactories) {
        return nullptr;
    }

    ccExternalFactory* factory = externalFactories->getFactoryByName(pluginId);
    if (!factory) {
        return nullptr;
    }

    ccHObject* obj = factory->buildObject(classId);

    if (name && obj) {
        obj->setName(name);
    }

    return obj;
}

QIcon ccHObject::getIcon() const { return QIcon(); }

/////////////////////// for python interface /////////////////////////////////
void ccHObject::ResizeAndPaintUniformColor(std::vector<Eigen::Vector3d>& colors,
                                           const size_t size,
                                           const Eigen::Vector3d& color) {
    colors.resize(size);
    Eigen::Vector3d clipped_color = color;
    if (color.minCoeff() < 0 || color.maxCoeff() > 1) {
        cloudViewer::utility::LogWarning(
                "[ccHObject::ResizeAndPaintUniformColor] invalid color in "
                "PaintUniformColor, clipping to [0, 1]");
        clipped_color = clipped_color.array()
                                .max(Eigen::Vector3d(0, 0, 0).array())
                                .matrix();
        clipped_color = clipped_color.array()
                                .min(Eigen::Vector3d(1, 1, 1).array())
                                .matrix();
    }
    for (size_t i = 0; i < size; i++) {
        colors[i] = clipped_color;
    }
}

Eigen::Vector3d ccHObject::ComputeMinBound(
        const std::vector<Eigen::Vector3d>& points) {
    if (points.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points.begin(), points.end(), points[0],
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a.array().min(b.array()).matrix();
            });
}

Eigen::Vector3d ccHObject::ComputeMaxBound(
        const std::vector<Eigen::Vector3d>& points) {
    if (points.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points.begin(), points.end(), points[0],
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a.array().max(b.array()).matrix();
            });
}

Eigen::Vector3d ccHObject::ComputeCenter(
        const std::vector<Eigen::Vector3d>& points) {
    Eigen::Vector3d center(0, 0, 0);
    if (points.empty()) {
        return center;
    }
    center = std::accumulate(points.begin(), points.end(), center);
    center /= double(points.size());
    return center;
}

void ccHObject::TransformPoints(const Eigen::Matrix4d& transformation,
                                std::vector<Eigen::Vector3d>& points) {
    for (auto& point : points) {
        Eigen::Vector4d new_point =
                transformation *
                Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.head<3>() / new_point(3);
    }
}

void ccHObject::TransformNormals(const Eigen::Matrix4d& transformation,
                                 std::vector<Eigen::Vector3d>& normals) {
    for (auto& normal : normals) {
        Eigen::Vector4d new_normal =
                transformation *
                Eigen::Vector4d(normal(0), normal(1), normal(2), 0.0);
        normal = new_normal.head<3>();
    }
}

void ccHObject::TransformCovariances(
        const Eigen::Matrix4d& transformation,
        std::vector<Eigen::Matrix3d>& covariances) {
    RotateCovariances(transformation.block<3, 3>(0, 0), covariances);
}

void ccHObject::TranslatePoints(const Eigen::Vector3d& translation,
                                std::vector<Eigen::Vector3d>& points,
                                bool relative) {
    Eigen::Vector3d transform = translation;
    if (!relative) {
        transform -= ComputeCenter(points);
    }
    for (auto& point : points) {
        point += transform;
    }
}

void ccHObject::ScalePoints(const double scale,
                            std::vector<Eigen::Vector3d>& points,
                            const Eigen::Vector3d& center) {
    for (auto& point : points) {
        point = (point - center) * scale + center;
    }
}

void ccHObject::RotatePoints(const Eigen::Matrix3d& R,
                             std::vector<Eigen::Vector3d>& points,
                             const Eigen::Vector3d& center) {
    for (auto& point : points) {
        point = R * (point - center) + center;
    }
}

void ccHObject::RotateNormals(const Eigen::Matrix3d& R,
                              std::vector<Eigen::Vector3d>& normals) {
    for (auto& normal : normals) {
        normal = R * normal;
    }
}

/// The only part that affects the covariance is the rotation part. For more
/// information on variance propagation please visit:
/// https://en.wikipedia.org/wiki/Propagation_of_uncertainty
void ccHObject::RotateCovariances(const Eigen::Matrix3d& R,
                                  std::vector<Eigen::Matrix3d>& covariances) {
    for (auto& covariance : covariances) {
        covariance = R * covariance * R.transpose();
    }
}

Eigen::Matrix3d ccHObject::GetRotationMatrixFromXYZ(
        const Eigen::Vector3d& rotation) {
    return cloudViewer::utility::RotationMatrixX(rotation(0)) *
           cloudViewer::utility::RotationMatrixY(rotation(1)) *
           cloudViewer::utility::RotationMatrixZ(rotation(2));
}

Eigen::Matrix3d ccHObject::GetRotationMatrixFromYZX(
        const Eigen::Vector3d& rotation) {
    return cloudViewer::utility::RotationMatrixY(rotation(0)) *
           cloudViewer::utility::RotationMatrixZ(rotation(1)) *
           cloudViewer::utility::RotationMatrixX(rotation(2));
}

Eigen::Matrix3d ccHObject::GetRotationMatrixFromZXY(
        const Eigen::Vector3d& rotation) {
    return cloudViewer::utility::RotationMatrixZ(rotation(0)) *
           cloudViewer::utility::RotationMatrixX(rotation(1)) *
           cloudViewer::utility::RotationMatrixY(rotation(2));
}

Eigen::Matrix3d ccHObject::GetRotationMatrixFromXZY(
        const Eigen::Vector3d& rotation) {
    return cloudViewer::utility::RotationMatrixX(rotation(0)) *
           cloudViewer::utility::RotationMatrixZ(rotation(1)) *
           cloudViewer::utility::RotationMatrixY(rotation(2));
}

Eigen::Matrix3d ccHObject::GetRotationMatrixFromZYX(
        const Eigen::Vector3d& rotation) {
    return cloudViewer::utility::RotationMatrixZ(rotation(0)) *
           cloudViewer::utility::RotationMatrixY(rotation(1)) *
           cloudViewer::utility::RotationMatrixX(rotation(2));
}

Eigen::Matrix3d ccHObject::GetRotationMatrixFromYXZ(
        const Eigen::Vector3d& rotation) {
    return cloudViewer::utility::RotationMatrixY(rotation(0)) *
           cloudViewer::utility::RotationMatrixX(rotation(1)) *
           cloudViewer::utility::RotationMatrixZ(rotation(2));
}

Eigen::Matrix3d ccHObject::GetRotationMatrixFromAxisAngle(
        const Eigen::Vector3d& rotation) {
    const double phi = rotation.norm();
    return Eigen::AngleAxisd(phi, rotation / phi).toRotationMatrix();
}

Eigen::Matrix3d ccHObject::GetRotationMatrixFromQuaternion(
        const Eigen::Vector4d& rotation) {
    return Eigen::Quaterniond(rotation(0), rotation(1), rotation(2),
                              rotation(3))
            .normalized()
            .toRotationMatrix();
}

Eigen::Matrix3d ccHObject::GetRotationMatrixFromEulerAngle(
        const Eigen::Vector3d& rotation) {
    Eigen::AngleAxisd rollAngle(
            Eigen::AngleAxisd(rotation(2), Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(
            Eigen::AngleAxisd(rotation(1), Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(
            Eigen::AngleAxisd(rotation(0), Eigen::Vector3d::UnitZ()));
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix = yawAngle * pitchAngle * rollAngle;
    return rotation_matrix;
}

ccBBox ccHObject::GetAxisAlignedBoundingBox() const { return ccBBox(); }

ecvOrientedBBox ccHObject::GetOrientedBoundingBox() const {
    return ecvOrientedBBox();
}

/////////////////////// for python interface /////////////////////////////////

void ccHObject::addDependency(ccHObject* otherObject,
                              int flags,
                              bool additive /*=true*/) {
    if (!otherObject || flags < 0) {
        CVLog::Error("[ccHObject::addDependency] Invalid arguments");
        assert(false);
        return;
    } else if (flags == 0) {
        return;
    }

    if (additive) {
        // look for already defined flags for this object
        std::map<ccHObject*, int>::iterator it =
                m_dependencies.find(otherObject);
        if (it != m_dependencies.end()) {
            // nothing changes? we stop here (especially to avoid infinite
            // loop when setting  the DP_NOTIFY_OTHER_ON_DELETE flag below!)
            if ((it->second & flags) == flags) return;
            flags |= it->second;
        }
    }
    assert(flags != 0);

    m_dependencies[otherObject] = flags;

    // whenever we add a dependency, we must be sure to be notified
    // by the other object when its deleted! Otherwise we'll keep
    // bad pointers in the dependency list...
    otherObject->addDependency(this, DP_NOTIFY_OTHER_ON_DELETE);
}

int ccHObject::getDependencyFlagsWith(const ccHObject* otherObject) {
    std::map<ccHObject*, int>::const_iterator it =
            m_dependencies.find(const_cast<ccHObject*>(
                    otherObject));  // DGM: not sure why erase won't accept a
                                    // const pointer?! We try to modify the map
                                    // here, not the pointer object!

    return (it != m_dependencies.end() ? it->second : 0);
}

void ccHObject::removeDependencyWith(ccHObject* otherObject) {
    m_dependencies.erase(const_cast<ccHObject*>(
            otherObject));  // DGM: not sure why erase won't accept a const
                            // pointer?! We try to modify the map here, not the
                            // pointer object!
    if (!otherObject->m_isDeleting)
        otherObject->removeDependencyFlag(this, DP_NOTIFY_OTHER_ON_DELETE);
}

void ccHObject::removeDependencyFlag(ccHObject* otherObject,
                                     DEPENDENCY_FLAGS flag) {
    int flags = getDependencyFlagsWith(otherObject);
    if ((flags & flag) == flag) {
        flags = (flags & (~flag));
        // either update the flags (if some bits remain)
        if (flags != 0)
            m_dependencies[otherObject] = flags;
        else  // otherwise remove the dependency
            m_dependencies.erase(otherObject);
    }
}

void ccHObject::onDeletionOf(const ccHObject* obj) {
    // remove any dependency declared with this object
    // and remove it from the children list as well (in case of)
    // DGM: we can't call 'detachChild' as this method will try to
    // modify the child contents!
    removeDependencyWith(const_cast<ccHObject*>(
            obj));  // this method will only modify the dependency flags of obj

    int pos = getChildIndex(obj);
    if (pos >= 0) {
        // we can't swap children as we want to keep the order!
        m_children.erase(m_children.begin() + pos);
    }
}

bool ccHObject::addChild(ccHObject* child,
                         int dependencyFlags /*=DP_PARENT_OF_OTHER*/,
                         int insertIndex /*=-1*/) {
    if (!child) {
        assert(false);
        return false;
    }
    if (std::find(m_children.begin(), m_children.end(), child) !=
        m_children.end()) {
        CVLog::ErrorDebug("[ccHObject::addChild] Object is already a child!");
        return false;
    }

    if (isLeaf()) {
        CVLog::ErrorDebug(
                "[ccHObject::addChild] Leaf objects shouldn't have any child!");
        return false;
    }

    // insert child
    try {
        if (insertIndex < 0 ||
            static_cast<size_t>(insertIndex) >= m_children.size())
            m_children.push_back(child);
        else
            m_children.insert(m_children.begin() + insertIndex, child);
    } catch (const std::bad_alloc&) {
        // not enough memory!
        return false;
    }

    // we want to be notified whenever this child is deleted!
    child->addDependency(
            this, DP_NOTIFY_OTHER_ON_DELETE);  // DGM: potentially redundant
                                               // with calls to 'addDependency'
                                               // but we can't miss that ;)

    if (dependencyFlags != 0) {
        addDependency(child, dependencyFlags);
    }

    // the strongest link: between a parent and a child ;)
    if ((dependencyFlags & DP_PARENT_OF_OTHER) == DP_PARENT_OF_OTHER) {
        child->setParent(this);
        if (child->isShareable()) dynamic_cast<CCShareable*>(child)->link();
    }

    return true;
}

unsigned int ccHObject::getChildCountRecursive() const {
    unsigned int count = static_cast<unsigned>(m_children.size());

    for (auto child : m_children) {
        count += child->getChildCountRecursive();
    }

    return count;
}

ccHObject* ccHObject::find(unsigned uniqueID) {
    // found the right item?
    if (getUniqueID() == uniqueID) {
        return this;
    }

    // otherwise we are going to test all children recursively
    for (unsigned i = 0; i < getChildrenNumber(); ++i) {
        ccHObject* match = getChild(i)->find(uniqueID);
        if (match) {
            return match;
        }
    }

    return nullptr;
}

unsigned ccHObject::filterChildren(Container& filteredChildren,
                                   bool recursive /*=false*/,
                                   CV_CLASS_ENUM filter /*=CV_TYPES::OBJECT*/,
                                   bool strict /*=false*/) const {
    for (auto child : m_children) {
        if ((!strict && child->isKindOf(filter)) ||
            (strict && child->isA(filter))) {
            // if (!inDisplay || child->getDisplay() == inDisplay)
            //{
            //  warning: we have to handle unicity as a sibling may be in the
            //  same container as its parent!
            if (std::find(filteredChildren.begin(), filteredChildren.end(),
                          child) ==
                filteredChildren.end())  // not yet in output vector?
            {
                filteredChildren.push_back(child);
            }
            //}
        }

        if (recursive) {
            child->filterChildren(filteredChildren, true, filter, strict);
        }
    }

    return static_cast<unsigned>(filteredChildren.size());
}

int ccHObject::getChildIndex(const ccHObject* child) const {
    for (size_t i = 0; i < m_children.size(); ++i)
        if (m_children[i] == child) return static_cast<int>(i);

    return -1;
}

void ccHObject::transferChild(ccHObject* child, ccHObject& newParent) {
    assert(child);

    // remove link from old parent
    int childDependencyFlags = child->getDependencyFlagsWith(this);
    int parentDependencyFlags = getDependencyFlagsWith(child);

    detachChild(child);  // automatically removes any dependency with this
                         // object

    newParent.addChild(child, parentDependencyFlags);
    child->addDependency(&newParent, childDependencyFlags);

    // after a successful transfer, either the parent is 'newParent' or a null
    // pointer
    assert(child->getParent() == &newParent || child->getParent() == nullptr);
}

void ccHObject::transferChildren(ccHObject& newParent,
                                 bool forceFatherDependent /*=false*/) {
    for (auto child : m_children) {
        // remove link from old parent
        int childDependencyFlags = child->getDependencyFlagsWith(this);
        int fatherDependencyFlags = getDependencyFlagsWith(child);

        // we must explicitely remove any dependency with the child as we don't
        // call 'detachChild'
        removeDependencyWith(child);
        child->removeDependencyWith(this);

        newParent.addChild(child, fatherDependencyFlags);
        child->addDependency(&newParent, childDependencyFlags);

        // after a successful transfer, either the parent is 'newParent' or a
        // null pointer
        assert(child->getParent() == &newParent ||
               child->getParent() == nullptr);
    }
    m_children.clear();
}

void ccHObject::swapChildren(unsigned firstChildIndex,
                             unsigned secondChildIndex) {
    assert(firstChildIndex < m_children.size());
    assert(secondChildIndex < m_children.size());

    std::swap(m_children[firstChildIndex], m_children[secondChildIndex]);
}

int ccHObject::getIndex() const {
    return (m_parent ? m_parent->getChildIndex(this) : -1);
}

bool ccHObject::isAncestorOf(const ccHObject* anObject) const {
    assert(anObject);
    ccHObject* parent = anObject->getParent();
    if (!parent) return false;

    if (parent == this) return true;

    return isAncestorOf(parent);
}

void ccHObject::removeFromRenderScreen(bool recursive) {
    CC_DRAW_CONTEXT context;
    context.removeViewID = getViewId();
    context.removeEntityType = getEntityType();
    context.display = mergeDisplay(nullptr, this);
    if (context.display) context.display->removeEntities(context);

    if (this->isKindOf(CV_TYPES::FACET) || this->isKindOf(CV_TYPES::PLANE)) {
        ccPlanarEntityInterface* plane = ccHObjectCaster::ToPlanarEntity(this);
        plane->showNormalVector(false);
        plane->clearNormalVector(context);
    }

    if (this->isKindOf(CV_TYPES::SENSOR)) {
        ccSensor* sensor = ccHObjectCaster::ToSensor(this);
        sensor->clearDrawings();
    }

    if (this->isKindOf(CV_TYPES::PRIMITIVE)) {
        ccGenericPrimitive* prim = ccHObjectCaster::ToPrimitive(this);
        if (prim) {
            prim->clearDrawings();
        }
    }

    if (recursive) {
        for (auto child : m_children) {
            child->removeFromRenderScreen(true);
        }
    }
}

bool ccHObject::getAbsoluteGLTransformation(ccGLMatrix& trans) const {
    trans.toIdentity();
    bool hasGLTrans = false;

    // recurse among ancestors to get the absolute GL transformation
    const ccHObject* obj = this;
    while (obj) {
        if (obj->isGLTransEnabled()) {
            trans = trans * obj->getGLTransformation();
            hasGLTrans = true;
        }
        obj = obj->getParent();
    }

    return hasGLTrans;
}

ccBBox ccHObject::getOwnBB(bool withGLFeatures /*=false*/) { return ccBBox(); }

ccBBox ccHObject::getBB_recursive(bool withGLFeatures /*=false*/,
                                  bool onlyEnabledChildren /*=true*/) {
    ccBBox box = getOwnBB(withGLFeatures);

    for (auto child : m_children) {
        if (!onlyEnabledChildren || child->isEnabled()) {
            box += child->getBB_recursive(withGLFeatures, onlyEnabledChildren);
        }
    }

    return box;
}

void ccHObject::setRedrawFlagRecursive(bool redraw /*=false*/) {
    // 2D Label or 2DLabel ViewPort
    if (isEnabled() || isKindOf(CV_TYPES::LABEL_2D)) {
        setRedraw(redraw);
    }

    for (auto child : m_children) {
        child->setRedrawFlagRecursive(redraw);
    }
}

void ccHObject::setForceRedrawRecursive(bool redraw) {
    setForceRedraw(redraw);

    for (auto child : m_children) {
        child->setForceRedrawRecursive(redraw);
    }
}

void ccHObject::setPointSizeRecursive(int pSize) {
    if (this->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(this);
        if (cloud && cloud->getPointSize() != pSize) {
            cloud->setPointSize(pSize);
        }
    }

    for (auto child : m_children) {
        child->setPointSizeRecursive(pSize);
    }
}

void ccHObject::setLineWidthRecursive(PointCoordinateType with) {
    if (this->isKindOf(CV_TYPES::POLY_LINE)) {
        ccPolyline* poly = ccHObjectCaster::ToPolyline(this);
        if (poly && poly->getWidth() != with) {
            poly->setWidth(with);
        }
    }

    for (auto child : m_children) {
        child->setLineWidthRecursive(with);
    }
}

ccBBox ccHObject::getDisplayBB_recursive(bool relative,
                                         const ecvGenericGLDisplay* display) {
    ccBBox box;
    // Only include this node's own BB if it belongs to the requested display
    // (or no display filter is set, or entity is unbound to any display).
    if (!display || m_currentDisplay == nullptr ||
        m_currentDisplay == display) {
        box = getOwnBB(true);
    }

    for (auto child : m_children) {
        if (child->isEnabled()) {
            ccBBox childBox = child->getDisplayBB_recursive(true, display);
            if (child->isGLTransEnabled()) {
                childBox = childBox * child->getGLTransformation();
            }
            box += childBox;
        }
    }

    if (!relative && box.isValid()) {
        ccGLMatrix trans;
        getAbsoluteGLTransformation(trans);
        box = box * trans;
    }

    return box;
}

void ccHObject::getTypeID_recursive(std::vector<removeInfo>& rmInfos,
                                    bool relative) {
    removeInfo rminfo;
    rminfo.removeId = getViewId();
    rminfo.removeType = getEntityType();
    if (rminfo.removeType == ENTITY_TYPE::ECV_OCTREE) {
        ccOctree* octree =
                ccHObjectCaster::ToOctree(find(rminfo.removeId.toUInt()));
        if (octree) {
            // remove temp octree model from rendering window
            octree->setVisible(false);
            CC_DRAW_CONTEXT context;
            octree->draw(context);
        }
    } else if (rminfo.removeType == ENTITY_TYPE::ECV_KDTREE) {
        ccKdTree* kdtree =
                ccHObjectCaster::ToKdTree(find(rminfo.removeId.toUInt()));
        if (kdtree) {
            // remove temp octree model from rendering window
            kdtree->setEnabled(false);
            CC_DRAW_CONTEXT context;
            kdtree->draw(context);
        }
    } else if (rminfo.removeType == ENTITY_TYPE::ECV_MESH) {
        ccHObject* obj = find(rminfo.removeId.toUInt());

        // try clear plane
        ccPlanarEntityInterface* plane = ccHObjectCaster::ToPlanarEntity(obj);
        if (plane) {
            // remove temp octree model from rendering window
            plane->showNormalVector(false);
            CC_DRAW_CONTEXT context;
            plane->glDrawNormal(context, CCVector3(), 1.0);
        }

        // try clear primitives
        ccGenericPrimitive* prim = ccHObjectCaster::ToPrimitive(obj);
        if (prim) {
            prim->clearDrawings();
        }
    } else if (rminfo.removeType == ENTITY_TYPE::ECV_2DLABLE) {
        ccHObject* obj = find(rminfo.removeId.toUInt());
        cc2DLabel* label = ccHObjectCaster::To2DLabel(obj);
        if (label) {
            // clear
            label->clearLabel(false);
        }
    } else if (rminfo.removeType == ENTITY_TYPE::ECV_2DLABLE_VIEWPORT) {
        ccHObject* obj = find(rminfo.removeId.toUInt());
        cc2DViewportLabel* labelViewPort =
                ccHObjectCaster::To2DViewportLabel(obj);
        if (labelViewPort) {
            // clear
            labelViewPort->clear2Dviews();
        }
    } else if (rminfo.removeType == ENTITY_TYPE::ECV_SENSOR) {
        ccHObject* obj = find(rminfo.removeId.toUInt());
        ccSensor* sensor = ccHObjectCaster::ToSensor(obj);
        if (sensor) {
            // clear
            sensor->clearDrawings();
        }
    }

    // need to remove 3D name if shown
    if (nameShownIn3D()) {
        showNameIn3D(false);
        WIDGETS_PARAMETER wp(WIDGETS_TYPE::WIDGET_T2D, getName());
        wp.context.display = mergeDisplay(nullptr, this);
        if (wp.context.display) wp.context.display->removeWidgets(wp);
    }
    rmInfos.push_back(rminfo);

    if (relative) {
        for (auto child : m_children) {
            child->getTypeID_recursive(rmInfos, true);
        }
    }
}

void ccHObject::getTypeID_recursive(std::vector<hideInfo>& hdInfos,
                                    bool relative) {
    hideInfo hdinfo;
    hdinfo.hideId = getViewId();
    hdinfo.hideType = getEntityType();
    hdInfos.push_back(hdinfo);

    if (relative) {
        for (auto child : m_children) {
            child->getTypeID_recursive(hdInfos, true);
        }
    }
}

bool ccHObject::isDisplayed() const { return isVisible() && isBranchEnabled(); }

bool ccHObject::isBranchEnabled() const {
    if (!isEnabled()) return false;

    if (m_parent) return m_parent->isBranchEnabled();

    return true;
}

void ccHObject::applyGLTransformation(const ccGLMatrix& trans) {
    m_glTransHistory = trans * m_glTransHistory;
}

void ccHObject::applyGLTransformation_recursive(
        const ccGLMatrix* transInput /*=nullptr*/) {
    ccGLMatrix transTemp;
    const ccGLMatrix* transToApply = transInput;

    if (m_glTransEnabled) {
        if (!transInput) {
            // if no transformation is provided (by father)
            // we initiate it with the current one
            transToApply = &m_glTrans;
        } else {
            transTemp = *transInput * m_glTrans;
            transToApply = &transTemp;
        }
    }

    if (transToApply) {
        applyGLTransformation(*transToApply);
        notifyGeometryUpdate();
    }

    for (auto child : m_children)
        child->applyGLTransformation_recursive(transToApply);

    if (m_glTransEnabled) resetGLTransformation();
}

unsigned ccHObject::findMaxUniqueID_recursive() const {
    unsigned id = getUniqueID();

    for (auto child : m_children) {
        unsigned childMaxID = child->findMaxUniqueID_recursive();
        if (id < childMaxID) {
            id = childMaxID;
        }
    }

    return id;
}

void ccHObject::detachChild(ccHObject* child) {
    if (!child) {
        assert(false);
        return;
    }

    // remove any dependency (bilateral)
    removeDependencyWith(child);
    child->removeDependencyWith(this);

    if (child->getParent() == this) {
        child->setParent(nullptr);
    }

    int pos = getChildIndex(child);
    if (pos >= 0) {
        // we can't swap children as we want to keep the order!
        m_children.erase(m_children.begin() + pos);
    }
}

ccBBox ccHObject::getOwnFitBB(ccGLMatrix& trans) {
    trans.toIdentity();
    return getOwnBB();
}

void ccHObject::drawBB(CC_DRAW_CONTEXT& context, const ecvColor::Rgb& col) {
    if (!ecvViewManager::instance().activeWidget()) {
        return;
    }

    switch (getSelectionBehavior()) {
        case SELECTION_AA_BBOX:
            getDisplayBB_recursive(true).draw(context, col);
            break;

        case SELECTION_FIT_BBOX: {
            ccGLMatrix trans;
            ccBBox box = getOwnFitBB(trans);
            if (box.isValid()) {
                ecvOrientedBBox obb =
                        ecvOrientedBBox::CreateFromAxisAlignedBoundingBox(box);
                obb.Transform(ccGLMatrixd::ToEigenMatrix4(trans));
                obb.draw(context, col);
            }
        } break;

        case SELECTION_IGNORED:
            break;

        default:
            assert(false);
    }
}

void ccHObject::detachAllChildren() {
    for (auto child : m_children) {
        // remove any dependency (bilateral)
        removeDependencyWith(child);
        child->removeDependencyWith(this);

        if (child->getParent() == this) {
            child->setParent(nullptr);
        }
    }
    m_children.clear();
}

void ccHObject::removeChild(ccHObject* child) {
    int pos = getChildIndex(child);
    if (pos >= 0) {
        removeChild(pos);
    }
}

void ccHObject::removeChild(int pos) {
    if (pos < 0 || static_cast<size_t>(pos) >= m_children.size()) {
        assert(false);
        return;
    }

    ccHObject* child = m_children[pos];

    // we can't swap as we want to keep the order!
    //(DGM: do this BEFORE deleting the object (otherwise
    // the dependency mechanism can 'backfire' ;)
    m_children.erase(m_children.begin() + pos);

    // backup dependency flags
    int flags = getDependencyFlagsWith(child);

    // remove any dependency
    removeDependencyWith(child);
    // child->removeDependencyWith(this); //DGM: no, don't do this otherwise
    // this entity won't be warned that the child has been removed!

    if ((flags & DP_DELETE_OTHER) == DP_DELETE_OTHER) {
        // delete object
        if (child->isShareable()) {
            dynamic_cast<CCShareable*>(child)->release();
        } else { /* if (!child->isA(CV_TYPES::POINT_OCTREE))*/
            delete child;
        }
    } else if (child->getParent() == this) {
        child->setParent(nullptr);
    }
}

void ccHObject::removeAllChildren() {
    while (!m_children.empty()) {
        ccHObject* child = m_children.back();
        m_children.pop_back();

        int flags = getDependencyFlagsWith(child);
        if ((flags & DP_DELETE_OTHER) == DP_DELETE_OTHER) {
            if (child->isShareable()) {
                dynamic_cast<CCShareable*>(child)->release();
            } else {
                delete child;
            }
        }
    }
}

bool ccHObject::isSerializable() const {
    // we only handle pure CV_TYPES::HIERARCHY_OBJECT here (object groups)
    return (getClassID() == CV_TYPES::HIERARCHY_OBJECT);
}

bool ccHObject::toFile(QFile& out, short dataVersion) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));

    // Version validation
    if (dataVersion < 23) {
        assert(false);
        return false;
    }

    // write 'ccObject' header
    if (!ccObject::toFile(out, dataVersion)) return false;

    // write own data
    if (!toFile_MeOnly(out, dataVersion)) return false;

    //(serializable) child count (dataVersion >= 20)
    uint32_t serializableCount = 0;
    for (auto child : m_children) {
        if (child->isSerializable()) {
            ++serializableCount;
        }
    }

    if (out.write(reinterpret_cast<const char*>(&serializableCount),
                  sizeof(uint32_t)) < 0)
        return WriteError();

    // write serializable children (if any)
    for (auto child : m_children) {
        if (child->isSerializable()) {
            if (!child->toFile(out, dataVersion)) return false;
        }
    }

    // write current selection behavior (dataVersion >= 23)
    if (out.write(reinterpret_cast<const char*>(&m_selectionBehavior),
                  sizeof(SelectionBehavior)) < 0)
        return WriteError();

    // write transformation history (dataVersion >= 45)
    if (dataVersion >= 45) {
        m_glTransHistory.toFile(out, dataVersion);
    }

    return true;
}

short ccHObject::minimumFileVersion() const {
    short minVersion = m_glTransHistory.isIdentity() ? 23 : 45;
    minVersion = std::max(minVersion, ccObject::minimumFileVersion());
    minVersion = std::max(minVersion, minimumFileVersion_MeOnly());

    // write serializable children (if any)
    for (auto child : m_children) {
        minVersion = std::max(minVersion, child->minimumFileVersion());
    }

    return minVersion;
}

bool ccHObject::fromFile(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) {
    if (!fromFileNoChildren(in, dataVersion, flags, oldToNewIDMap))
        return false;

    //(serializable) child count (dataVersion>=20)
    uint32_t serializableCount = 0;
    if (in.read(reinterpret_cast<char*>(&serializableCount), 4) < 0)
        return ReadError();

    // read serializable children (if any)
    for (uint32_t i = 0; i < serializableCount; ++i) {
        // read children class ID
        CV_CLASS_ENUM classID = ReadClassIDFromFile(in, dataVersion);
        if (classID == CV_TYPES::OBJECT) return false;

        if (dataVersion >= 35 && dataVersion <= 47 &&
            ((classID & CC_CUSTOM_BIT) != 0)) {
            // bug fix: for a long time the CC_CAMERA_BIT and CC_QUADRIC_BIT
            // were wrongly defined with two bits instead of one! The additional
            // and wrongly defined bit was the CC_CUSTOM_BIT :(
            if ((classID & CV_TYPES::CAMERA_SENSOR) ==
                        CV_TYPES::CAMERA_SENSOR ||
                (classID & CV_TYPES::QUADRIC) == CV_TYPES::QUADRIC) {
                classID &= (~CC_CUSTOM_BIT);
            }
        }

        // create corresponding child object
        ccHObject* child = New(classID);

        // specific case of custom objects (defined by plugins)
        if ((classID & CV_TYPES::CUSTOM_H_OBJECT) ==
            CV_TYPES::CUSTOM_H_OBJECT) {
            // store current position
            size_t originalFilePos = in.pos();
            // we need to load the custom object as plain ccCustomHobject
            child->fromFileNoChildren(in, dataVersion, flags, oldToNewIDMap);
            // go back to original position
            in.seek(originalFilePos);
            // get custom object name and plugin name
            QString childName = child->getName();
            QString classId =
                    child->getMetaData(
                                 ccCustomHObject::DefautMetaDataClassName())
                            .toString();
            QString pluginId =
                    child->getMetaData(
                                 ccCustomHObject::DefautMetaDataPluginName())
                            .toString();
            // dont' need this instance anymore
            delete child;
            child = nullptr;

            // try to get a new object from external factories
            ccHObject* newChild = ccHObject::New(pluginId, classId);
            if (newChild)  // found a plugin that can deserialize it
            {
                child = newChild;
            } else {
                CVLog::Warning(
                        QString("[ccHObject::fromFile] Couldn't find a plugin "
                                "able to deserialize custom object '%1' "
                                "(class_ID = %2 / plugin_ID = %3)")
                                .arg(childName)
                                .arg(classID)
                                .arg(pluginId));
                return false;  // FIXME: for now simply return false. We may
                               // want to skip it but I'm not sure if there is a
                               // simple way of doing that
            }
        }

        assert(child && child->isSerializable());
        if (child) {
            if (child->fromFile(in, dataVersion, flags, oldToNewIDMap)) {
                // FIXME
                // addChild(child,child->getFlagState(CC_FATHER_DEPENDENT));
                addChild(child);
            } else {
                // delete child; //we can't do this as the object might be
                // invalid
                return false;
            }
        } else {
            return CorruptError();
        }
    }

    // read the selection behavior (dataVersion>=23)
    if (dataVersion >= 23) {
        if (in.read(reinterpret_cast<char*>(&m_selectionBehavior),
                    sizeof(SelectionBehavior)) < 0) {
            return ReadError();
        }
    } else {
        m_selectionBehavior = SELECTION_AA_BBOX;
    }

    // read transformation history (dataVersion >= 45)
    if (dataVersion >= 45) {
        if (!m_glTransHistory.fromFile(in, dataVersion, flags, oldToNewIDMap)) {
            return false;
        }
    }

    return true;
}

bool ccHObject::fromFileNoChildren(QFile& in,
                                   short dataVersion,
                                   int flags,
                                   LoadedIDMap& oldToNewIDMap) {
    assert(in.isOpen() && (in.openMode() & QIODevice::ReadOnly));

    // read 'ccObject' header
    if (!ccObject::fromFile(in, dataVersion, flags, oldToNewIDMap))
        return false;

    // read own data
    return fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap);
}

bool ccHObject::toFile_MeOnly(QFile& out, short dataVersion) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));

    // Version validation
    if (dataVersion < 20) {
        assert(false);
        return false;
    }

    /*** ccHObject takes in charge the ccDrawableObject properties (which is not
     * a ccSerializableObject) ***/

    //'visible' state (dataVersion>=20)
    if (out.write(reinterpret_cast<const char*>(&m_visible), sizeof(bool)) < 0)
        return WriteError();
    //'lockedVisibility' state (dataVersion>=20)
    if (out.write(reinterpret_cast<const char*>(&m_lockedVisibility),
                  sizeof(bool)) < 0)
        return WriteError();
    //'colorsDisplayed' state (dataVersion>=20)
    if (out.write(reinterpret_cast<const char*>(&m_colorsDisplayed),
                  sizeof(bool)) < 0)
        return WriteError();
    //'normalsDisplayed' state (dataVersion>=20)
    if (out.write(reinterpret_cast<const char*>(&m_normalsDisplayed),
                  sizeof(bool)) < 0)
        return WriteError();
    //'sfDisplayed' state (dataVersion>=20)
    if (out.write(reinterpret_cast<const char*>(&m_sfDisplayed), sizeof(bool)) <
        0)
        return WriteError();
    //'colorIsOverridden' state (dataVersion>=20)
    if (out.write(reinterpret_cast<const char*>(&m_colorIsOverridden),
                  sizeof(bool)) < 0)
        return WriteError();
    if (m_colorIsOverridden) {
        //'tempColor' (dataVersion>=20)
        if (out.write(reinterpret_cast<const char*>(m_tempColor.rgb),
                      sizeof(ColorCompType) * 3) < 0) {
            return WriteError();
        }
    }
    //'glTransEnabled' state (dataVersion>=20)
    if (out.write(reinterpret_cast<const char*>(&m_glTransEnabled),
                  sizeof(bool)) < 0)
        return WriteError();
    if (m_glTransEnabled) {
        if (!m_glTrans.toFile(out, dataVersion)) {
            return false;
        }
    }

    //'showNameIn3D' state (dataVersion>=24)
    if (dataVersion >= 24) {
        if (out.write(reinterpret_cast<const char*>(&m_showNameIn3D),
                      sizeof(bool)) < 0)
            return WriteError();
    }

    return true;
}

short ccHObject::minimumFileVersion_MeOnly() const {
    // Determine minimum version based on feature usage:
    // - Version 20: Basic drawable properties
    // - Version 24: showNameIn3D state
    return m_showNameIn3D ? 24 : 20;
}

bool ccHObject::fromFile_MeOnly(QFile& in,
                                short dataVersion,
                                int flags,
                                LoadedIDMap& oldToNewIDMap) {
    assert(in.isOpen() && (in.openMode() & QIODevice::ReadOnly));

    /*** ccHObject takes in charge the ccDrawableObject properties (which is not
     * a ccSerializableObject) ***/

    //'visible' state (dataVersion>=20)
    if (in.read(reinterpret_cast<char*>(&m_visible), sizeof(bool)) < 0)
        return ReadError();
    //'lockedVisibility' state (dataVersion>=20)
    if (in.read(reinterpret_cast<char*>(&m_lockedVisibility), sizeof(bool)) < 0)
        return ReadError();
    //'colorsDisplayed' state (dataVersion>=20)
    if (in.read(reinterpret_cast<char*>(&m_colorsDisplayed), sizeof(bool)) < 0)
        return ReadError();
    //'normalsDisplayed' state (dataVersion>=20)
    if (in.read(reinterpret_cast<char*>(&m_normalsDisplayed), sizeof(bool)) < 0)
        return ReadError();
    //'sfDisplayed' state (dataVersion>=20)
    if (in.read(reinterpret_cast<char*>(&m_sfDisplayed), sizeof(bool)) < 0)
        return ReadError();
    //'colorIsOverriden' state (dataVersion>=20)
    if (in.read(reinterpret_cast<char*>(&m_colorIsOverridden), sizeof(bool)) <
        0)
        return ReadError();
    if (m_colorIsOverridden) {
        //'tempColor' (dataVersion>=20)
        if (in.read(reinterpret_cast<char*>(m_tempColor.rgb),
                    sizeof(ColorCompType) * 3) < 0)
            return ReadError();
    }
    //'glTransEnabled' state (dataVersion>=20)
    if (in.read(reinterpret_cast<char*>(&m_glTransEnabled), sizeof(bool)) < 0)
        return ReadError();
    if (m_glTransEnabled) {
        if (!m_glTrans.fromFile(in, dataVersion, flags, oldToNewIDMap)) {
            m_glTransEnabled = false;
            return false;
        }
    }

    //'showNameIn3D' state (dataVersion>=24)
    if (dataVersion >= 24) {
        if (in.read(reinterpret_cast<char*>(&m_showNameIn3D), sizeof(bool)) <
            0) {
            return ReadError();
        }
    } else {
        m_showNameIn3D = false;
    }

    return true;
}

void ccHObject::drawNameIn3D() {
    ecvGenericGLDisplay* disp = mergeDisplay(nullptr, this);
    QFont font = QApplication::font();
    const ecvGui::ParamStruct* fontParams = nullptr;
    if (disp) {
        fontParams = &disp->getDisplayParameters();
    } else if (ecvGenericGLDisplay* ev =
                       ecvViewManager::instance().getEffectiveView()) {
        fontParams = &ev->getDisplayParameters();
    }
    if (fontParams) {
        font.setPointSize(static_cast<int>(fontParams->defaultFontSize));
    } else {
        font.setPointSize(
                static_cast<int>(ecvGui::Parameters().defaultFontSize));
    }
    if (disp) {
        disp->display2DText(getName(), static_cast<int>(m_nameIn3DPos.x),
                            static_cast<int>(m_nameIn3DPos.y),
                            static_cast<unsigned char>(
                                    ecvGenericDisplayTools::ALIGN_HMIDDLE |
                                    ecvGenericDisplayTools::ALIGN_VMIDDLE),
                            0.75f, nullptr, &font, getViewId());
    } else if (ecvGenericGLDisplay* ev =
                       ecvViewManager::instance().getEffectiveView()) {
        ev->display2DText(getName(), static_cast<int>(m_nameIn3DPos.x),
                          static_cast<int>(m_nameIn3DPos.y),
                          static_cast<unsigned char>(
                                  ecvGenericDisplayTools::ALIGN_HMIDDLE |
                                  ecvGenericDisplayTools::ALIGN_VMIDDLE),
                          0.75f, nullptr, &font, getViewId());
    }
}

void ccHObject::draw(CC_DRAW_CONTEXT& context) {
    // for polyline fast removement
    if (getRemoveFlag()) {
        setRemoveType(context);
        context.removeViewID = getViewId();
        context.display = mergeDisplay(context.display, this);
        if (context.display) context.display->removeEntities(context);
        return;
    }

    // are we currently drawing objects in 2D or 3D?
    bool draw3D = MACRO_Draw3D(context);

    if (!isEnabled() && !isKindOf(CV_TYPES::POINT_OCTREE) &&
        !isKindOf(CV_TYPES::POINT_KDTREE)) {
        hideObject_recursive(true);
        return;
    }

    // the entity must be either visible or selected, and of course it should be
    // displayed in this context
    bool drawInThisContext =
            ((m_visible || m_selected) && isDisplayedIn(context.display));

    ecvViewRepresentation* viewRep = nullptr;
    const bool isLabel2D = isKindOf(CV_TYPES::LABEL_2D);
    const bool isHierarchy = (getClassID() == CV_TYPES::HIERARCHY_OBJECT);
    // Only ensure (create) a per-view representation for entities that are
    // truly visible -- not for entities that are only drawn because they are
    // selected.  ensureRepresentation() inherits donor visibility which can
    // incorrectly cascade a "hidden" state from another view.
    // HIERARCHY_OBJECT (folders) are pure containers with no VTK geometry;
    // they must NOT get per-view reps or they will incorrectly block
    // children's visibility through the cascade.
    const bool trueVisible =
            m_visible && isDisplayedIn(context.display);
    if (context.display && trueVisible && !isFixedId() && !isLabel2D &&
        !isHierarchy) {
        viewRep = ecvRepresentationManager::instance().ensureRepresentation(
                const_cast<ccHObject*>(this), context.display);
    } else if (context.display && !isHierarchy) {
        viewRep = ecvRepresentationManager::instance().getRepresentation(
                const_cast<ccHObject*>(this), context.display);
    }
    if (drawInThisContext && viewRep && viewRep->hasVisibilityOverride()) {
        drawInThisContext = viewRep->isVisible();
    }

    const bool ancestorVisible = context.visible;
    if (isHierarchy) {
        context.visible = ancestorVisible;
    } else if (viewRep && viewRep->hasVisibilityOverride()) {
        context.visible = viewRep->isVisible() && ancestorVisible;
    } else if (drawInThisContext) {
        context.visible = m_visible && ancestorVisible;
    } else {
        context.visible = ancestorVisible;
    }
    const bool cascadeVisible = context.visible;

    context.opacity = (viewRep && viewRep->properties().opacity.has_value())
                              ? viewRep->effectiveOpacity()
                              : getOpacity();

    if (viewRep && viewRep->properties().pointSize.has_value()) {
        context.defaultPointSize =
                static_cast<unsigned char>(viewRep->effectivePointSize());
    }
    if (viewRep && viewRep->properties().lineWidth.has_value()) {
        context.defaultLineWidth =
                static_cast<unsigned char>(viewRep->effectiveLineWidth());
        context.currentLineWidth = context.defaultLineWidth;
    }
    if (viewRep && viewRep->properties().renderMode.has_value()) {
        auto rm = viewRep->effectiveRenderMode();
        if (rm != ecvViewRepresentation::RenderMode::Inherit) {
            context.meshRenderingMode =
                    static_cast<MESH_RENDERING_MODE>(static_cast<int>(rm));
        }
    }
    if (viewRep && viewRep->properties().normalScale.has_value()) {
        context.normalScale = viewRep->effectiveNormalScale();
    }

    if (!isFixedId()) {
        context.viewID = getViewId();
    }

    if (draw3D) {
        // apply 3D 'temporary' transformation (for display only)
        if (m_glTransEnabled) {
            // context.transformInfo.setRotMat(m_glTrans);
        }

        // LOD for clouds is enabled?
        if (context.decimateCloudOnMove && context.currentLODLevel > 0) {
            // only for real clouds
            drawInThisContext &= isA(CV_TYPES::POINT_CLOUD);
        }
    }

    // draw entity
    if (m_visible && drawInThisContext && context.forceRedraw) {
        if ((!m_selected || !MACRO_SkipSelected(context)) &&
            (m_selected || !MACRO_SkipUnselected(context))) {
            // enable clipping planes (if any)
            bool useClipPlanes = (draw3D && !m_clipPlanes.empty());
            if (useClipPlanes) {
                toggleClipPlanes(context, true);
            }

            drawMeOnly(context);

            if (viewRep && viewRep->isDirty()) {
                viewRep->setDirty(false);
            }

            // disable clipping planes (if any)
            if (useClipPlanes) {
                toggleClipPlanes(context, false);
            }
        }
    }

    // hide or show entities
    {
        setHideShowType(context);
        context.display = mergeDisplay(context.display, this);
        bool hasExist = true;
        if (context.display) {
            hasExist = context.display->hideShowEntities(context);
        }
        if (!context.forceRedraw && m_forceRedraw && !hasExist) {
            if (!isA(CV_TYPES::OBJECT) && !isA(CV_TYPES::HIERARCHY_OBJECT) &&
                !isA(CV_TYPES::TRANS_BUFFER)) {
                setForceRedraw(false);
                CC_DRAW_CONTEXT newContext = context;
                newContext.forceRedraw = true;
                setRedrawFlagRecursive(true);
                draw(newContext);
            }
        }
    }

    // draw name - only when the object (or its parent hierarchy) is actually
    // visible/enabled.  Container objects (folders) that are enabled but not
    // "visible" in the geometric sense can still show a name, but objects
    // whose visibility checkbox is unchecked, or whose parent folder is
    // disabled, must not draw names.
    // Guard with isDisplayedIn() to prevent accessing the wrong VTK pipeline
    // when the object belongs to a secondary view but the primary redraws.
    bool shouldDrawName = m_showNameIn3D && !MACRO_EntityPicking(context) &&
                          isDisplayedIn(context.display) &&
                          (m_visible || m_selected);
    if (shouldDrawName) {
        if (MACRO_Draw3D(context)) {
            ccBBox bBox = getBB_recursive(true);
            if (bBox.isValid()) {
                ccGLCameraParameters camera;
                ecvGenericGLDisplay* camDisp =
                        mergeDisplay(context.display, this);
                if (camDisp) {
                    camDisp->getGLCameraParameters(camera);
                } else if (auto* ev = ecvViewManager::instance()
                                              .getEffectiveView()) {
                    ev->getGLCameraParameters(camera);
                }

                CCVector3 C = bBox.getCenter();
                camera.project(C, m_nameIn3DPos);
            }
        } else if (MACRO_Draw2D(context) && MACRO_Foreground(context)) {
            drawNameIn3D();
        }
    } else if (!shouldDrawName && isDisplayedIn(context.display)) {
        if (!isKindOf(CV_TYPES::LABEL_2D)) {
            WIDGETS_PARAMETER wpTxt(WIDGETS_TYPE::WIDGET_T2D, getName());
            wpTxt.context.display = mergeDisplay(context.display, this);
            if (wpTxt.context.display)
                wpTxt.context.display->removeWidgets(wpTxt);
            WIDGETS_PARAMETER wpRect(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                                     getName());
            wpRect.context.display = mergeDisplay(context.display, this);
            if (wpRect.context.display)
                wpRect.context.display->removeWidgets(wpRect);
        }
    }

    // draw entity's children
    for (auto child : m_children) {
        context.visible = cascadeVisible;
        child->draw(context);
    }

    // if the entity is currently selected, we draw its bounding-box
    if (m_selected && draw3D && drawInThisContext &&
        !MACRO_EntityPicking(context) && context.currentLODLevel == 0) {
        // Check if BoundingBox should be shown
        const ecvGui::ParamStruct& params = ecvGui::Parameters();
        bool shouldShowBB = params.showBBOnSelected;

        // Check if Axes Grid is visible - if so, ALWAYS hide BoundingBox
        // (unconditionally, regardless of showBBOnSelected setting)
        if (ecvGenericGLDisplay* axesDisp =
                    mergeDisplay(context.display, this)) {
            AxesGridProperties axesGridProps;
            axesDisp->getDataAxesGridProperties(context.viewID, axesGridProps);
            if (axesGridProps.visible) {
                shouldShowBB =
                        false;  // Force hide BBox when axes grid is visible
            }
        }

        if (shouldShowBB) {
            CC_DRAW_CONTEXT tempContext = context;
            tempContext.meshRenderingMode =
                    MESH_RENDERING_MODE::ECV_WIREFRAME_MODE;
            tempContext.viewID = getViewId();
            // Apply BoundingBox color, opacity and line width from parameters
            tempContext.bbDefaultCol = params.bbDefaultCol;
            tempContext.opacity = params.bbOpacity;
            tempContext.defaultLineWidth =
                    static_cast<unsigned char>(params.bbLineWidth);
            tempContext.currentLineWidth = tempContext.defaultLineWidth;
            drawBB(tempContext, params.bbDefaultCol);
            tempContext.viewID = getViewId();
            showBB(tempContext);
        } else {
            // Hide BoundingBox if not should show
            CC_DRAW_CONTEXT tempContext = context;
            tempContext.viewID = getViewId();
            hideBB(tempContext);
        }
    }

    if (!m_selected && draw3D) {
        CC_DRAW_CONTEXT context;
        context.viewID = getViewId();
        hideBB(context);
    }

    // reset redraw flag to true and forceRedraw flag to false(default)
    setRedraw(true);
    setForceRedraw(false);
}

void ccHObject::updateNameIn3DRecursive() {
    if (nameShownIn3D() && isEnabled() && (isVisible() || isSelected())) {
        ecvGenericGLDisplay* disp = mergeDisplay(nullptr, this);

        ccBBox bBox = getBB_recursive(true);
        if (bBox.isValid()) {
            ccGLCameraParameters camera;
            if (disp) {
                disp->getGLCameraParameters(camera);
            } else if (auto* ev =
                               ecvViewManager::instance().getEffectiveView()) {
                ev->getGLCameraParameters(camera);
            }

            CCVector3 C = bBox.getCenter();
            camera.project(C, m_nameIn3DPos);

            WIDGETS_PARAMETER wp(WIDGETS_TYPE::WIDGET_T2D, getViewId());
            wp.context.display = disp;
            if (wp.context.display) wp.context.display->removeWidgets(wp);

            drawNameIn3D();
        }
    }

    for (auto child : m_children) {
        child->updateNameIn3DRecursive();
    }
}

void ccHObject::setHideShowType(CC_DRAW_CONTEXT& context) {
    context.hideShowEntityType = convertClassToEntityType(getClassID());
}

ENTITY_TYPE ccHObject::getEntityType() const {
    return convertClassToEntityType(getClassID());
}

void ccHObject::setRemoveType(CC_DRAW_CONTEXT& context) {
    context.removeEntityType = convertClassToEntityType(getClassID());
}

void ccHObject::hideBB(CC_DRAW_CONTEXT context) {
    context.hideShowEntityType = ENTITY_TYPE::ECV_SHAPE;
    context.viewID = QString("BBox-") + context.viewID;
    context.visible = false;
    context.display = mergeDisplay(context.display, this);
    if (context.display) context.display->hideShowEntities(context);
}

void ccHObject::showBB(CC_DRAW_CONTEXT context) {
    context.hideShowEntityType = ENTITY_TYPE::ECV_SHAPE;
    context.viewID = QString("BBox-") + context.viewID;
    context.visible = true;
    context.display = mergeDisplay(context.display, this);
    if (context.display) context.display->hideShowEntities(context);
}

ccHObject::GlobalBoundingBox ccHObject::getOwnGlobalBB(
        bool withGLFeatures /*=false*/) {
    // by default this method returns the local bounding-box!
    ccBBox box = getOwnBB(false);
    return GlobalBoundingBox(CCVector3d::fromArray(box.minCorner().u),
                             CCVector3d::fromArray(box.maxCorner().u));
}

bool ccHObject::getOwnGlobalBB(CCVector3d& minCorner, CCVector3d& maxCorner) {
    // by default this method returns the local bounding-box!
    ccBBox box = getOwnBB(false);
    minCorner = CCVector3d::fromArray(box.minCorner().u);
    maxCorner = CCVector3d::fromArray(box.maxCorner().u);
    return box.isValid();
}

ccHObject::GlobalBoundingBox ccHObject::getGlobalBB_recursive(
        bool withGLFeatures /*=false*/, bool onlyEnabledChildren /*=true*/) {
    GlobalBoundingBox box = getOwnGlobalBB(withGLFeatures);

    for (auto child : m_children) {
        if (!onlyEnabledChildren || child->isEnabled()) {
            box += child->getGlobalBB_recursive(withGLFeatures,
                                                onlyEnabledChildren);
        }
    }

    return box;
}

void ccHObject::hideObject_recursive(bool recursive) {
    std::vector<hideInfo> hdInfos;
    CC_DRAW_CONTEXT context;
    getTypeID_recursive(hdInfos, recursive);
    context.visible = false;
    context.display = mergeDisplay(nullptr, this);

    // Helper: propagate hideShowEntities to the primary display and, for
    // unbound entities, to every registered view so actors don't remain
    // visible in non-active windows.
    auto hideInAllViews = [&](CC_DRAW_CONTEXT& ctx) {
        if (ctx.display) ctx.display->hideShowEntities(ctx);
        if (!getDisplay()) {
            for (auto* view : ecvViewManager::instance().getAllViews()) {
                if (!view || view == ctx.display) continue;
                CC_DRAW_CONTEXT vctx = ctx;
                vctx.display = view;
                view->hideShowEntities(vctx);
            }
        }
    };

    for (const hideInfo& hdInfo : hdInfos) {
        if (hdInfo.hideType == ENTITY_TYPE::ECV_NONE) continue;

        context.hideShowEntityType = hdInfo.hideType;
        context.viewID = hdInfo.hideId;
        hideBB(context);
        context.viewID = hdInfo.hideId;

        ccHObject* obj = find(hdInfo.hideId.toUInt());

        if (hdInfo.hideType == ENTITY_TYPE::ECV_2DLABLE ||
            hdInfo.hideType == ENTITY_TYPE::ECV_2DLABLE_VIEWPORT) {
            context.viewID = hdInfo.hideId;
            hideInAllViews(context);
            if (hdInfo.hideType == ENTITY_TYPE::ECV_2DLABLE) {
                cc2DLabel* label = dynamic_cast<cc2DLabel*>(obj);
                if (label) {
                    label->clearLabel(true);
                }
            }
            continue;
        } else if (hdInfo.hideType == ENTITY_TYPE::ECV_SENSOR) {
            ccSensor* sensor = ccHObjectCaster::ToSensor(obj);
            if (sensor) {
                sensor->hideShowDrawings(context);
                continue;
            }
        } else if (hdInfo.hideType == ENTITY_TYPE::ECV_MESH) {
            ccGenericPrimitive* prim = ccHObjectCaster::ToPrimitive(obj);
            if (prim) {
                prim->hideShowDrawings(context);
            }
            if (obj) {
                if (obj->isKindOf(CV_TYPES::FACET)) {
                    static_cast<ccFacet*>(obj)->hideNormalArrowActors(context);
                } else if (obj->isKindOf(CV_TYPES::PLANE)) {
                    static_cast<ccPlane*>(obj)->hideNormalArrowActors(context);
                }
            }
            if (prim) continue;
        } else if (hdInfo.hideType == ENTITY_TYPE::ECV_OCTREE ||
                   hdInfo.hideType == ENTITY_TYPE::ECV_KDTREE) {
            ccOctreeProxy* proxy = ccHObjectCaster::ToOctreeProxy(obj);
            if (proxy && proxy->getOctree()) {
                proxy->setOctreeVisibale(false);
                CC_DRAW_CONTEXT octCtx = context;
                octCtx.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
                octCtx.forceRedraw = true;
                proxy->getOctree()->draw(octCtx);
            }
            continue;
        }

        context.viewID = hdInfo.hideId;
        hideInAllViews(context);
    }
}

void ccHObject::toggleVisibility_recursive(bool visible, bool recursive) {
    std::vector<hideInfo> hdInfos;
    CC_DRAW_CONTEXT context;
    getTypeID_recursive(hdInfos, recursive);
    context.visible = visible;
    context.display = mergeDisplay(nullptr, this);
    for (const hideInfo& hdInfo : hdInfos) {
        if (hdInfo.hideType == ENTITY_TYPE::ECV_NONE) continue;

        if ((hdInfo.hideType == ENTITY_TYPE::ECV_OCTREE ||
             hdInfo.hideType == ENTITY_TYPE::ECV_KDTREE)) {
            ccHObject* obj = find(hdInfo.hideId.toUInt());
            ccOctreeProxy* proxy = ccHObjectCaster::ToOctreeProxy(obj);
            if (proxy && proxy->getOctree()) {
                proxy->setOctreeVisibale(visible);
                CC_DRAW_CONTEXT octCtx = context;
                octCtx.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
                octCtx.forceRedraw = true;
                proxy->getOctree()->draw(octCtx);
            }
            continue;
        }

        context.hideShowEntityType = hdInfo.hideType;
        context.viewID = hdInfo.hideId;
        if (context.display) context.display->hideShowEntities(context);
        if (!getDisplay()) {
            for (auto* view : ecvViewManager::instance().getAllViews()) {
                if (!view || view == context.display) continue;
                CC_DRAW_CONTEXT vctx = context;
                vctx.display = view;
                view->hideShowEntities(vctx);
            }
        }

        if (!visible) {
            hideBB(context);
            context.viewID = hdInfo.hideId;
            ccHObject* obj = find(hdInfo.hideId.toUInt());
            if (obj) {
                if (obj->isKindOf(CV_TYPES::FACET)) {
                    static_cast<ccFacet*>(obj)->hideNormalArrowActors(context);
                } else if (obj->isKindOf(CV_TYPES::PLANE)) {
                    static_cast<ccPlane*>(obj)->hideNormalArrowActors(context);
                }
            }
        } else {
            ccHObject* obj = find(hdInfo.hideId.toUInt());
            if (obj) {
                if (obj->isKindOf(CV_TYPES::FACET)) {
                    static_cast<ccFacet*>(obj)->showNormalArrowActors(context);
                } else if (obj->isKindOf(CV_TYPES::PLANE)) {
                    static_cast<ccPlane*>(obj)->showNormalArrowActors(context);
                }
            }
        }
    }
}

void ccHObject::redrawDisplay(bool forceRedraw /* = true*/,
                              bool only2D /* = false*/) {
    if (m_currentDisplay) {
        m_currentDisplay->redraw(only2D, forceRedraw);
        auto* effective = ecvViewManager::instance().getEffectiveView();
        if (effective && effective != m_currentDisplay) {
            ecvRedrawScope scope(only2D, forceRedraw);
        }
    } else {
        ecvRedrawScope scope(only2D, forceRedraw);
    }
}

struct HObjectDisplayState : ccDrawableObject::DisplayState {
    HObjectDisplayState() {}

    HObjectDisplayState(const ccHObject& obj)
        : ccDrawableObject::DisplayState(obj), isEnabled(obj.isEnabled()) {}

    bool isEnabled = false;
};

bool ccHObject::pushDisplayState() {
    try {
        m_displayStateStack.emplace_back(new HObjectDisplayState(*this));
    } catch (const std::bad_alloc&) {
        CVLog::Warning("Not enough memory to push the current display state");
        return false;
    }

    return true;
}

void ccHObject::popDisplayState(bool apply /*=true*/) {
    if (!m_displayStateStack.empty()) {
        const DisplayState::Shared state = m_displayStateStack.back();
        if (state && apply) {
            HObjectDisplayState* hState =
                    static_cast<HObjectDisplayState*>(state.data());
            if (hState->isEnabled != isEnabled()) {
                setEnabled(hState->isEnabled);
            }
            applyDisplayState(*state);
        }
        m_displayStateStack.pop_back();
    }
}

void ccHObject::removeFromDisplay_recursive(
        const ecvGenericGLDisplay* display) {
    removeFromDisplay(display);
    for (auto child : m_children) {
        child->removeFromDisplay_recursive(display);
    }
}

bool ccHObject::isDisplayedIn(const ecvGenericGLDisplay* display) const {
    if (display == nullptr) return true;

    if (m_currentDisplay == nullptr) {
        // Unbound objects (no setDisplay() called) render in ALL views.
        // This is the standard multi-window behavior: point clouds, meshes,
        // and other 3D objects should be visible across all split views.
        // Per-view restriction is achieved by calling setDisplay() on
        // objects that should only appear in one view (e.g., cc2DLabel).
        return true;
    }
    return (m_currentDisplay == display);
}
