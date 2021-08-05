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

#include "ecvHObject.h"
#include "ecvBBox.h"

//Objects handled by factory
#include "ecv2DLabel.h"
#include "ecv2DViewportLabel.h"
#include "ecvBox.h"
#include "ecvCustomObject.h"
#include "ecvCylinder.h"
#include "ecvCoordinateSystem.h"
#include "ecvPolyline.h"
#include "ecvFacet.h"
#include "ecvImage.h"
#include "ecvGBLSensor.h"
#include "ecvMaterialSet.h"
#include "ecvCameraSensor.h"
#include "ecvPointCloud.h"
#include "ecvExtru.h"
#include "ecvMeshGroup.h"
#include "ecvPlane.h"
#include "ecvQuadric.h"
#include "ecvSphere.h"
#include "ecvSubMesh.h"
#include "ecvTorus.h"
#include "ecvKdTree.h"
#include "ecvDisplayTools.h"
#include "ecvDish.h"
#include "ecvHObjectCaster.h"
#include "ecvExternalFactory.h"
#include "ecvIndexedTransformationBuffer.h"

// CV_CORE_LIB
#include <Eigen.h>
#include <CVTools.h>
#include <Logging.h>

#include <Eigen/Dense>
#include <numeric>

// Qt
#include <QIcon>

ccHObject::ccHObject(QString name/*=QString()*/)
	: ccObject(name)
	, ccDrawableObject()
	, m_parent(nullptr)
	, m_selectionBehavior(SELECTION_AA_BBOX)
	, m_isDeleting(false)
{
	setVisible(false);
	lockVisibility(true);
	m_glTransHistory.toIdentity();
}

ccHObject::ccHObject(const ccHObject& object)
	: ccObject(object)
	, ccDrawableObject(object)
	, m_parent(nullptr)
	, m_selectionBehavior(object.m_selectionBehavior)
	, m_isDeleting(false)
{
	m_glTransHistory.toIdentity();
}

ccHObject::~ccHObject()
{
	m_isDeleting = true;

	//process dependencies
	for (std::map<ccHObject*, int>::const_iterator it = m_dependencies.begin(); it != m_dependencies.end(); ++it)
	{
		assert(it->first);
		//notify deletion to other object?
		if ((it->second & DP_NOTIFY_OTHER_ON_DELETE) == DP_NOTIFY_OTHER_ON_DELETE)
		{
			it->first->onDeletionOf(this);
		}

		//delete other object?
		if ((it->second & DP_DELETE_OTHER) == DP_DELETE_OTHER)
		{
			it->first->removeDependencyFlag(this, DP_NOTIFY_OTHER_ON_DELETE); //in order to avoid any loop!
			//delete object
			if (it->first->isShareable())
				dynamic_cast<CCShareable*>(it->first)->release();
			else
				delete it->first;
		}
	}
	m_dependencies.clear();

	removeAllChildren();
}

void ccHObject::notifyGeometryUpdate()
{
	//the associated display bounding-box is (potentially) deprecated!!!
	if (!ecvDisplayTools::TheInstance())
	{
		return;
	}

	if (ecvDisplayTools::GetCurrentScreen())
	{
		ecvDisplayTools::InvalidateViewport();
		ecvDisplayTools::Deprecate3DLayer();
        ecvDisplayTools::RemoveBB(getViewId());
	}

	//process dependencies
	for (std::map<ccHObject*, int>::const_iterator it = m_dependencies.begin(); it != m_dependencies.end(); ++it)
	{
		assert(it->first);
		//notify deletion to other object?
		if ((it->second & DP_NOTIFY_OTHER_ON_UPDATE) == DP_NOTIFY_OTHER_ON_UPDATE)
		{
			it->first->onUpdateOf(this);
		}
	}
}

ccHObject* ccHObject::New(CV_CLASS_ENUM objectType, const char* name/*=0*/)
{
	switch (objectType)
	{
	case CV_TYPES::HIERARCHY_OBJECT:
		return new ccHObject(name);
	case CV_TYPES::POINT_CLOUD:
		return new ccPointCloud(name);
	case CV_TYPES::MESH:
		//warning: no associated vertices --> retrieved later
		return new ccMesh(nullptr);
	case CV_TYPES::SUB_MESH:
		//warning: no associated mesh --> retrieved later
		return new ccSubMesh(nullptr);
	case CV_TYPES::MESH_GROUP:
		//warning: deprecated
		CVLog::Warning("[ccHObject::New] Mesh groups are deprecated!");
		//warning: no associated vertices --> retrieved later
		return new ccMeshGroup();
	case CV_TYPES::POLY_LINE:
		//warning: no associated vertices --> retrieved later
		return new ccPolyline(nullptr);
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
		return nullptr; //deprecated
	case CV_TYPES::GBL_SENSOR:
		//warning: default sensor type set in constructor (see CCLib::GroundBasedLidarSensor::setRotationOrder)
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
		//construction this way is not supported (yet)
		CVLog::ErrorDebug("[ccHObject::New] This object (type %i) can't be constructed this way (yet)!", objectType);
		break;
	default:
		//unhandled ID
		CVLog::ErrorDebug("[ccHObject::New] Invalid object type (%i)!", objectType);
		break;
	}

	return nullptr;
}

ccHObject* ccHObject::New(const QString& pluginId, const QString& classId, const char* name)
{
	ccExternalFactory::Container::Shared externalFactories = ccExternalFactory::Container::GetUniqueInstance();
	if (!externalFactories)
	{
		return nullptr;
	}

	ccExternalFactory* factory = externalFactories->getFactoryByName(pluginId);
	if (!factory)
	{
		return nullptr;
	}

	ccHObject* obj = factory->buildObject(classId);

	if (name && obj)
	{
		obj->setName(name);
	}

	return obj;
}

QIcon ccHObject::getIcon() const
{
    return QIcon();
}

/////////////////////// for python interface /////////////////////////////////
void ccHObject::ResizeAndPaintUniformColor(
	std::vector<Eigen::Vector3d>& colors,
	const size_t size,
	const Eigen::Vector3d& color) {
    colors.resize(size);
    Eigen::Vector3d clipped_color = color;
    if (color.minCoeff() < 0 || color.maxCoeff() > 1) {
        cloudViewer::utility::LogWarning(
                "[ccHObject::ResizeAndPaintUniformColor] invalid color in paintUniformColor, clipping to [0, 1]");
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
	const std::vector<Eigen::Vector3d>& points)
{
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
	const std::vector<Eigen::Vector3d>& points)
{
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

void ccHObject::TransformPoints(
	const Eigen::Matrix4d& transformation,
	std::vector<Eigen::Vector3d>& points) {
    for (auto& point : points) {
        Eigen::Vector4d new_point =
                transformation *
                Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.head<3>() / new_point(3);
    }
}

void ccHObject::TransformNormals(
	const Eigen::Matrix4d& transformation,
	std::vector<Eigen::Vector3d>& normals)
{
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

void ccHObject::TranslatePoints(
	const Eigen::Vector3d& translation,
	std::vector<Eigen::Vector3d>& points,
	bool relative)
{
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
	const Eigen::Vector3d& center)
{
    for (auto& point : points) {
        point = (point - center) * scale + center;
    }
}

void ccHObject::RotatePoints(const Eigen::Matrix3d& R,
	std::vector<Eigen::Vector3d>& points,
	const Eigen::Vector3d& center)
{
    for (auto& point : points) {
        point = R * (point - center) + center;
    }
}

void ccHObject::RotateNormals(const Eigen::Matrix3d& R,
	std::vector<Eigen::Vector3d>& normals)
{
    for (auto& normal : normals) {
        normal = R * normal;
    }
}

/// The only part that affects the covariance is the rotation part. For more
/// information on variance propagation please visit:
/// https://en.wikipedia.org/wiki/Propagation_of_uncertainty
void ccHObject::RotateCovariances(
        const Eigen::Matrix3d& R,
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

Eigen::Matrix3d ccHObject::GetRotationMatrixFromEulerAngle(const Eigen::Vector3d& rotation)
{
	Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(rotation(2), Eigen::Vector3d::UnitX()));
	Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(rotation(1), Eigen::Vector3d::UnitY()));
	Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(rotation(0), Eigen::Vector3d::UnitZ()));
	Eigen::Matrix3d rotation_matrix;
	rotation_matrix = yawAngle * pitchAngle * rollAngle;
	return rotation_matrix;
}

ccBBox ccHObject::getAxisAlignedBoundingBox() const
{
	return ccBBox();
}

ecvOrientedBBox ccHObject::getOrientedBoundingBox() const
{
	return ecvOrientedBBox();
}

/////////////////////// for python interface /////////////////////////////////

void ccHObject::addDependency(ccHObject* otherObject, int flags, bool additive/*=true*/)
{
	if (!otherObject || flags < 0)
	{
		CVLog::Error("[ccHObject::addDependency] Invalid arguments");
		assert(false);
		return;
	}
	else if (flags == 0)
	{
		return;
	}

	if (additive)
	{
		//look for already defined flags for this object
		std::map<ccHObject*, int>::iterator it = m_dependencies.find(otherObject);
		if (it != m_dependencies.end())
		{
			//nothing changes? we stop here (especially to avoid infinite
			//loop when setting  the DP_NOTIFY_OTHER_ON_DELETE flag below!)
			if ((it->second & flags) == flags)
				return;
			flags |= it->second;
		}
	}
	assert(flags != 0);

	m_dependencies[otherObject] = flags;

	//whenever we add a dependency, we must be sure to be notified
	//by the other object when its deleted! Otherwise we'll keep
	//bad pointers in the dependency list...
	otherObject->addDependency(this, DP_NOTIFY_OTHER_ON_DELETE);
}

int ccHObject::getDependencyFlagsWith(const ccHObject* otherObject)
{
	std::map<ccHObject*, int>::const_iterator it = m_dependencies.find(const_cast<ccHObject*>(otherObject)); //DGM: not sure why erase won't accept a const pointer?! We try to modify the map here, not the pointer object!

	return (it != m_dependencies.end() ? it->second : 0);
}

void ccHObject::removeDependencyWith(ccHObject* otherObject)
{
	m_dependencies.erase(const_cast<ccHObject*>(otherObject)); //DGM: not sure why erase won't accept a const pointer?! We try to modify the map here, not the pointer object!
	if (!otherObject->m_isDeleting)
		otherObject->removeDependencyFlag(this, DP_NOTIFY_OTHER_ON_DELETE);
}

void ccHObject::removeDependencyFlag(ccHObject* otherObject, DEPENDENCY_FLAGS flag)
{
	int flags = getDependencyFlagsWith(otherObject);
	if ((flags & flag) == flag)
	{
		flags = (flags & (~flag));
		//either update the flags (if some bits remain)
		if (flags != 0)
			m_dependencies[otherObject] = flags;
		else //otherwise remove the dependency
			m_dependencies.erase(otherObject);
	}
}

void ccHObject::onDeletionOf(const ccHObject* obj)
{
	//remove any dependency declared with this object
	//and remove it from the children list as well (in case of)
	//DGM: we can't call 'detachChild' as this method will try to
	//modify the child contents!
	removeDependencyWith(const_cast<ccHObject*>(obj)); //this method will only modify the dependency flags of obj

	int pos = getChildIndex(obj);
	if (pos >= 0)
	{
		//we can't swap children as we want to keep the order!
		m_children.erase(m_children.begin() + pos);
	}
}

bool ccHObject::addChild(ccHObject* child, int dependencyFlags/*=DP_PARENT_OF_OTHER*/, int insertIndex/*=-1*/)
{
	if (!child)
	{
		assert(false);
		return false;
	}
	if (std::find(m_children.begin(), m_children.end(), child) != m_children.end())
	{
		CVLog::ErrorDebug("[ccHObject::addChild] Object is already a child!");
		return false;
	}

	if (isLeaf())
	{
		CVLog::ErrorDebug("[ccHObject::addChild] Leaf objects shouldn't have any child!");
		return false;
	}

	//insert child
	try
	{
		if (insertIndex < 0 || static_cast<size_t>(insertIndex) >= m_children.size())
			m_children.push_back(child);
		else
			m_children.insert(m_children.begin() + insertIndex, child);
	}
	catch (const std::bad_alloc&)
	{
		//not enough memory!
		return false;
	}

	//we want to be notified whenever this child is deleted!
	child->addDependency(this, DP_NOTIFY_OTHER_ON_DELETE); //DGM: potentially redundant with calls to 'addDependency' but we can't miss that ;)

	if (dependencyFlags != 0)
	{
		addDependency(child, dependencyFlags);
	}

	//the strongest link: between a parent and a child ;)
	if ((dependencyFlags & DP_PARENT_OF_OTHER) == DP_PARENT_OF_OTHER)
	{
		child->setParent(this);
		if (child->isShareable())
			dynamic_cast<CCShareable*>(child)->link();
	}

	return true;
}

unsigned int ccHObject::getChildCountRecursive() const
{
	unsigned int	count = static_cast<unsigned>(m_children.size());

	for (auto child : m_children)
	{
		count += child->getChildCountRecursive();
	}

	return count;
}

ccHObject* ccHObject::find(unsigned uniqueID)
{
	//found the right item?
	if (getUniqueID() == uniqueID)
	{
		return this;
	}

	//otherwise we are going to test all children recursively
	for (unsigned i = 0; i < getChildrenNumber(); ++i)
	{
		ccHObject* match = getChild(i)->find(uniqueID);
		if (match)
		{
			return match;
		}
	}

	return nullptr;
}

unsigned ccHObject::filterChildren(Container& filteredChildren,
	bool recursive/*=false*/,
	CV_CLASS_ENUM filter/*=CV_TYPES::OBJECT*/,
	bool strict/*=false*/) const
{
	for (auto child : m_children)
	{
		if ((!strict && child->isKindOf(filter))
			|| (strict && child->isA(filter)))
		{
			//if (!inDisplay || child->getDisplay() == inDisplay)
			//{
			// warning: we have to handle unicity as a sibling may be in the same container as its parent!
			if (std::find(filteredChildren.begin(), filteredChildren.end(), child) == filteredChildren.end()) //not yet in output vector?
			{
				filteredChildren.push_back(child);
			}
			//}

		}

		if (recursive)
		{
			child->filterChildren(filteredChildren, true, filter, strict);
		}
	}

	return static_cast<unsigned>(filteredChildren.size());
}


int ccHObject::getChildIndex(const ccHObject* child) const
{
	for (size_t i = 0; i < m_children.size(); ++i)
		if (m_children[i] == child)
			return static_cast<int>(i);

	return -1;
}

void ccHObject::transferChild(ccHObject* child, ccHObject& newParent)
{
	assert(child);

	//remove link from old parent
	int childDependencyFlags = child->getDependencyFlagsWith(this);
	int parentDependencyFlags = getDependencyFlagsWith(child);

	detachChild(child); //automatically removes any dependency with this object

	newParent.addChild(child, parentDependencyFlags);
	child->addDependency(&newParent, childDependencyFlags);

	//after a successful transfer, either the parent is 'newParent' or a null pointer
	assert(child->getParent() == &newParent || child->getParent() == nullptr);
}

void ccHObject::transferChildren(ccHObject& newParent, bool forceFatherDependent/*=false*/)
{
	for (auto child : m_children)
	{
		//remove link from old parent
		int childDependencyFlags = child->getDependencyFlagsWith(this);
		int fatherDependencyFlags = getDependencyFlagsWith(child);

		//we must explicitely remove any dependency with the child as we don't call 'detachChild'
		removeDependencyWith(child);
		child->removeDependencyWith(this);

		newParent.addChild(child, fatherDependencyFlags);
		child->addDependency(&newParent, childDependencyFlags);

		//after a successful transfer, either the parent is 'newParent' or a null pointer
		assert(child->getParent() == &newParent || child->getParent() == nullptr);
	}
	m_children.clear();
}

void ccHObject::swapChildren(unsigned firstChildIndex, unsigned secondChildIndex)
{
	assert(firstChildIndex < m_children.size());
	assert(secondChildIndex < m_children.size());

	std::swap(m_children[firstChildIndex], m_children[secondChildIndex]);
}

int ccHObject::getIndex() const
{
	return (m_parent ? m_parent->getChildIndex(this) : -1);
}

bool ccHObject::isAncestorOf(const ccHObject *anObject) const
{
	assert(anObject);
	ccHObject* parent = anObject->getParent();
	if (!parent)
		return false;

	if (parent == this)
		return true;

	return isAncestorOf(parent);
}

void ccHObject::removeFromRenderScreen(bool recursive)
{
	CC_DRAW_CONTEXT context;
    context.removeViewID = getViewId();
    context.removeEntityType = getEntityType();
	ecvDisplayTools::RemoveEntities(context);

	if (this->isKindOf(CV_TYPES::FACET) || this->isKindOf(CV_TYPES::PLANE))
	{
		ccPlanarEntityInterface* plane = ccHObjectCaster::ToPlanarEntity(this);
		plane->showNormalVector(false);
        plane->clearNormalVector(context);
	}

    if (this->isKindOf(CV_TYPES::SENSOR))
    {
        ccSensor* sensor = ccHObjectCaster::ToSensor(this);
        sensor->clearDrawings();
    }

    if (this->isKindOf(CV_TYPES::PRIMITIVE))
    {
        ccGenericPrimitive* prim = ccHObjectCaster::ToPrimitive(this);
        if (prim)
        {
            prim->clearDrawings();
        }
    }

	if (recursive)
	{
		for (auto child : m_children)
		{
			child->removeFromRenderScreen(true);
		}
	}
}

bool ccHObject::getAbsoluteGLTransformation(ccGLMatrix& trans) const
{
	trans.toIdentity();
	bool hasGLTrans = false;

	//recurse among ancestors to get the absolute GL transformation
	const ccHObject* obj = this;
	while (obj)
	{
		if (obj->isGLTransEnabled())
		{
			trans = trans * obj->getGLTransformation();
			hasGLTrans = true;
		}
		obj = obj->getParent();
	}

	return hasGLTrans;
}

ccBBox ccHObject::getOwnBB(bool withGLFeatures/*=false*/)
{
	return ccBBox();
}

ccBBox ccHObject::getBB_recursive(bool withGLFeatures/*=false*/, bool onlyEnabledChildren/*=true*/)
{
	ccBBox box = getOwnBB(withGLFeatures);

	for (auto child : m_children)
	{
		if (!onlyEnabledChildren || child->isEnabled())
		{
			box += child->getBB_recursive(withGLFeatures, onlyEnabledChildren);
		}
	}

	return box;
}

void ccHObject::setRedrawFlagRecursive(bool redraw/*=false*/)
{
	// 2D Label or 2DLabel ViewPort
	if (isEnabled() || isKindOf(CV_TYPES::LABEL_2D))
	{
		setRedraw(redraw);
	}
	
	for (auto child : m_children)
	{
		child->setRedrawFlagRecursive(redraw);
	}
}

void ccHObject::setForceRedrawRecursive(bool redraw)
{
	setForceRedraw(redraw);

	for (auto child : m_children)
	{
		child->setForceRedrawRecursive(redraw);
	}
}

void ccHObject::setPointSizeRecursive(int pSize)
{
	if (this->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(this);
		if (cloud && cloud->getPointSize() != pSize)
		{
			cloud->setPointSize(pSize);
		}
	}

	for (auto child : m_children)
	{
		child->setPointSizeRecursive(pSize);
	}
}

void ccHObject::setLineWidthRecursive(PointCoordinateType with)
{
	if (this->isKindOf(CV_TYPES::POLY_LINE))
	{
		ccPolyline* poly = ccHObjectCaster::ToPolyline(this);
		if (poly && poly->getWidth() != with)
		{
			poly->setWidth(with);
		}
	}

	for (auto child : m_children)
	{
		child->setLineWidthRecursive(with);
	}
}

ccBBox ccHObject::getDisplayBB_recursive(bool relative)
{
	ccBBox box;
	box = getOwnBB(true);

	for (auto child : m_children)
	{
		if (child->isEnabled())
		{
			ccBBox childBox = child->getDisplayBB_recursive(true);
			if (child->isGLTransEnabled())
			{
				childBox = childBox * child->getGLTransformation();
			}
			box += childBox;
		}
	}

	if (!relative && box.isValid())
	{
		//get absolute bounding-box?
		ccGLMatrix trans;
		getAbsoluteGLTransformation(trans);
		box = box * trans;
	}

	return box;
}

void ccHObject::getTypeID_recursive(std::vector<removeInfo> & rmInfos, bool relative)
{
	removeInfo rminfo;
    rminfo.removeId = getViewId();
	rminfo.removeType = getEntityType();
	if (rminfo.removeType == ENTITY_TYPE::ECV_OCTREE)
	{
		ccOctree* octree = ccHObjectCaster::ToOctree(find(rminfo.removeId.toUInt()));
		if (octree)
		{
			// remove temp octree model from rendering window
			octree->setVisible(false);
            CC_DRAW_CONTEXT context;
            octree->draw(context);
		}
	}
	else if (rminfo.removeType == ENTITY_TYPE::ECV_KDTREE)
	{
		ccKdTree* kdtree = ccHObjectCaster::ToKdTree(find(rminfo.removeId.toUInt()));
		if (kdtree)
		{
			// remove temp octree model from rendering window
			kdtree->setEnabled(false);
            CC_DRAW_CONTEXT context;
            kdtree->draw(context);
		}
	}
	else if (rminfo.removeType == ENTITY_TYPE::ECV_MESH)
	{
		ccHObject* obj = find(rminfo.removeId.toUInt());

        // try clear plane
		ccPlanarEntityInterface* plane = ccHObjectCaster::ToPlanarEntity(obj);
		if (plane)
		{
			// remove temp octree model from rendering window
			plane->showNormalVector(false);
            CC_DRAW_CONTEXT context;
            plane->glDrawNormal(context, CCVector3(), 1.0);
		}

        // try clear primitives
        ccGenericPrimitive* prim = ccHObjectCaster::ToPrimitive(obj);
        if (prim)
        {
            prim->clearDrawings();
        }
	}
	else if (rminfo.removeType == ENTITY_TYPE::ECV_2DLABLE)
	{
		ccHObject* obj = find(rminfo.removeId.toUInt());
		cc2DLabel* label = ccHObjectCaster::To2DLabel(obj);
		if (label)
		{
			// clear
			label->clearLabel(false);
		}
	}
	else if (rminfo.removeType == ENTITY_TYPE::ECV_2DLABLE_VIEWPORT)
	{
		ccHObject* obj = find(rminfo.removeId.toUInt());
		cc2DViewportLabel* labelViewPort = ccHObjectCaster::To2DViewportLabel(obj);
		if (labelViewPort)
		{
			// clear
			labelViewPort->clear2Dviews();
		}
    }
    else if (rminfo.removeType == ENTITY_TYPE::ECV_SENSOR)
    {
        ccHObject* obj = find(rminfo.removeId.toUInt());
        ccSensor* sensor = ccHObjectCaster::ToSensor(obj);
        if (sensor)
        {
            // clear
            sensor->clearDrawings();
        }
    }

	// need to remove 3D name if shown
	if (nameShownIn3D())
	{
		showNameIn3D(false);
		ecvDisplayTools::RemoveWidgets(
			WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, getName()));
	}
	rmInfos.push_back(rminfo);

	if (relative) {
		for (auto child : m_children)
		{
			child->getTypeID_recursive(rmInfos, true);
		}
	}
}

void ccHObject::getTypeID_recursive(std::vector<hideInfo> & hdInfos, bool relative)
{
	hideInfo hdinfo;
    hdinfo.hideId = getViewId();
	hdinfo.hideType = getEntityType();
	hdInfos.push_back(hdinfo);

	if (relative)
	{
		for (auto child : m_children)
		{
			child->getTypeID_recursive(hdInfos, true);
		}
	}
}

bool ccHObject::isDisplayed() const
{
	return isVisible() && isBranchEnabled();
}

bool ccHObject::isBranchEnabled() const
{
	if (!isEnabled())
		return false;

	if (m_parent)
		return m_parent->isBranchEnabled();

	return true;
}

void ccHObject::applyGLTransformation(const ccGLMatrix& trans)
{
	m_glTransHistory = trans * m_glTransHistory;
}

void ccHObject::applyGLTransformation_recursive(const ccGLMatrix* transInput/*=nullptr*/)
{
	ccGLMatrix transTemp;
	const ccGLMatrix* transToApply = transInput;

	if (m_glTransEnabled)
	{
		if (!transInput)
		{
			//if no transformation is provided (by father)
			//we initiate it with the current one
			transToApply = &m_glTrans;
		}
		else
		{
			transTemp = *transInput * m_glTrans;
			transToApply = &transTemp;
		}
	}

	if (transToApply)
	{
		applyGLTransformation(*transToApply);
		notifyGeometryUpdate();
	}

	for (auto child : m_children)
		child->applyGLTransformation_recursive(transToApply);

	if (m_glTransEnabled)
		resetGLTransformation();
}

unsigned ccHObject::findMaxUniqueID_recursive() const
{
	unsigned id = getUniqueID();

	for (auto child : m_children)
	{
		unsigned childMaxID = child->findMaxUniqueID_recursive();
		if (id < childMaxID)
		{
			id = childMaxID;
		}
	}

	return id;
}

void ccHObject::detachChild(ccHObject* child)
{
	if (!child)
	{
		assert(false);
		return;
	}

	//remove any dependency (bilateral)
	removeDependencyWith(child);
	child->removeDependencyWith(this);

	if (child->getParent() == this)
	{
		child->setParent(nullptr);
	}

	int pos = getChildIndex(child);
	if (pos >= 0)
	{
		//we can't swap children as we want to keep the order!
		m_children.erase(m_children.begin() + pos);
	}
}

ccBBox ccHObject::getOwnFitBB(ccGLMatrix & trans)
{
	trans.toIdentity(); 
	return getOwnBB();
}

void ccHObject::drawBB(CC_DRAW_CONTEXT& context, const ecvColor::Rgb& col)
{
	if (!ecvDisplayTools::GetMainWindow())
	{
		return;
	}

	switch (getSelectionBehavior())
	{
	case SELECTION_AA_BBOX:
		getDisplayBB_recursive(true).draw(context, col);
		break;

	case SELECTION_FIT_BBOX:
	{
		ccGLMatrix trans;
		ccBBox box = getOwnFitBB(trans);
		if (box.isValid())
        {
            ecvOrientedBBox obb = ecvOrientedBBox::CreateFromAxisAlignedBoundingBox(box);
            obb.transform(ccGLMatrixd::ToEigenMatrix4(trans));
			obb.draw(context, col);
		}
	}
	break;

	case SELECTION_IGNORED:
		break;

	default:
		assert(false);
	}
}

void ccHObject::detatchAllChildren()
{
	for (auto child : m_children)
	{
		//remove any dependency (bilateral)
		removeDependencyWith(child);
		child->removeDependencyWith(this);

		if (child->getParent() == this)
		{
			child->setParent(nullptr);
		}
	}
	m_children.clear();
}

void ccHObject::removeChild(ccHObject* child)
{
	int pos = getChildIndex(child);
	if (pos >= 0)
	{
		removeChild(pos);
	}
}

void ccHObject::removeChild(int pos)
{
	if (pos < 0 || static_cast<size_t>(pos) >= m_children.size())
	{
		assert(false);
		return;
	}

	ccHObject* child = m_children[pos];

	//we can't swap as we want to keep the order!
	//(DGM: do this BEFORE deleting the object (otherwise
	//the dependency mechanism can 'backfire' ;)
	m_children.erase(m_children.begin() + pos);

	//backup dependency flags
	int flags = getDependencyFlagsWith(child);

	//remove any dependency
	removeDependencyWith(child);
	//child->removeDependencyWith(this); //DGM: no, don't do this otherwise this entity won't be warned that the child has been removed!

	if ((flags & DP_DELETE_OTHER) == DP_DELETE_OTHER)
	{
		//delete object
		if (child->isShareable())
			dynamic_cast<CCShareable*>(child)->release();
		else/* if (!child->isA(CV_TYPES::POINT_OCTREE))*/
			delete child;
	}
	else if (child->getParent() == this)
	{
		child->setParent(nullptr);
	}
}

void ccHObject::removeAllChildren()
{
	while (!m_children.empty())
	{
		ccHObject* child = m_children.back();
		m_children.pop_back();

		int flags = getDependencyFlagsWith(child);
		if ((flags & DP_DELETE_OTHER) == DP_DELETE_OTHER)
		{
			if (child->isShareable())
				dynamic_cast<CCShareable*>(child)->release();
			else
				delete child;
		}
	}
}

bool ccHObject::isSerializable() const
{
	//we only handle pure CV_TYPES::HIERARCHY_OBJECT here (object groups)
	return (getClassID() == CV_TYPES::HIERARCHY_OBJECT);
}

bool ccHObject::toFile(QFile& out) const
{
	assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));

	//write 'ccObject' header
	if (!ccObject::toFile(out))
		return false;

	//write own data
	if (!toFile_MeOnly(out))
		return false;

	//(serializable) child count (dataVersion >= 20)
	uint32_t serializableCount = 0;
	for (auto child : m_children)
	{
		if (child->isSerializable())
		{
			++serializableCount;
		}
	}

	if (out.write(reinterpret_cast<const char*>(&serializableCount), sizeof(uint32_t)) < 0)
		return WriteError();

	//write serializable children (if any)
	for (auto child : m_children)
	{
		if (child->isSerializable())
		{
			if (!child->toFile(out))
				return false;
		}
	}

	//write current selection behavior (dataVersion >= 23)
	if (out.write(reinterpret_cast<const char*>(&m_selectionBehavior), sizeof(SelectionBehavior)) < 0)
		return WriteError();

	//write transformation history (dataVersion >= 45)
	m_glTransHistory.toFile(out);

	return true;
}

bool ccHObject::fromFile(QFile& in, short dataVersion, int flags, LoadedIDMap& oldToNewIDMap)
{
    if (!fromFileNoChildren(in, dataVersion, flags, oldToNewIDMap))
		return false;

	//(serializable) child count (dataVersion>=20)
	uint32_t serializableCount = 0;
	if (in.read(reinterpret_cast<char*>(&serializableCount), 4) < 0)
		return ReadError();

	//read serializable children (if any)
	for (uint32_t i = 0; i < serializableCount; ++i)
	{
		//read children class ID
		CV_CLASS_ENUM classID = ReadClassIDFromFile(in, dataVersion);
		if (classID == CV_TYPES::OBJECT)
			return false;

		if (dataVersion >= 35 && dataVersion <= 47 && ((classID & CC_CUSTOM_BIT) != 0))
		{
			//bug fix: for a long time the CC_CAMERA_BIT and CC_QUADRIC_BIT were wrongly defined
			//with two bits instead of one! The additional and wrongly defined bit was the CC_CUSTOM_BIT :(
			if ((classID & CV_TYPES::CAMERA_SENSOR) == CV_TYPES::CAMERA_SENSOR
				|| (classID & CV_TYPES::QUADRIC) == CV_TYPES::QUADRIC
				)
			{
				classID &= (~CC_CUSTOM_BIT);
			}
		}

		//create corresponding child object
		ccHObject* child = New(classID);

		//specific case of custom objects (defined by plugins)
		if ((classID & CV_TYPES::CUSTOM_H_OBJECT) == CV_TYPES::CUSTOM_H_OBJECT)
		{
			//store current position
			size_t originalFilePos = in.pos();
			//we need to load the custom object as plain ccCustomHobject
            child->fromFileNoChildren(in, dataVersion, flags, oldToNewIDMap);
			//go back to original position
			in.seek(originalFilePos);
			//get custom object name and plugin name
			QString childName = child->getName();
			QString classId = child->getMetaData(ccCustomHObject::DefautMetaDataClassName()).toString();
			QString pluginId = child->getMetaData(ccCustomHObject::DefautMetaDataPluginName()).toString();
			//dont' need this instance anymore
			delete child;
			child = nullptr;

			// try to get a new object from external factories
			ccHObject* newChild = ccHObject::New(pluginId, classId);
			if (newChild) // found a plugin that can deserialize it
			{
				child = newChild;
			}
			else
			{
				CVLog::Warning(QString("[ccHObject::fromFile] Couldn't find a plugin able to deserialize custom object '%1' (class_ID = %2 / plugin_ID = %3)").arg(childName).arg(classID).arg(pluginId));
				return false; // FIXME: for now simply return false. We may want to skip it but I'm not sure if there is a simple way of doing that
			}
		}

		assert(child && child->isSerializable());
		if (child)
		{
            if (child->fromFile(in, dataVersion, flags, oldToNewIDMap))
			{
				//FIXME
				//addChild(child,child->getFlagState(CC_FATHER_DEPENDENT));
				addChild(child);
			}
			else
			{
				//delete child; //we can't do this as the object might be invalid
				return false;
			}
		}
		else
		{
			return CorruptError();
		}
	}

	//read the selection behavior (dataVersion>=23)
	if (dataVersion >= 23)
	{
		if (in.read(reinterpret_cast<char*>(&m_selectionBehavior), sizeof(SelectionBehavior)) < 0)
		{
			return ReadError();
		}
	}
	else
	{
		m_selectionBehavior = SELECTION_AA_BBOX;
	}

	//read transformation history (dataVersion >= 45)
	if (dataVersion >= 45)
	{
        if (!m_glTransHistory.fromFile(in, dataVersion, flags, oldToNewIDMap))
		{
			return false;
		}
	}

	return true;
}

bool ccHObject::fromFileNoChildren(QFile& in, short dataVersion, int flags, LoadedIDMap& oldToNewIDMap)
{
	assert(in.isOpen() && (in.openMode() & QIODevice::ReadOnly));

	//read 'ccObject' header
    if (!ccObject::fromFile(in, dataVersion, flags, oldToNewIDMap))
		return false;

	//read own data
    return fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap);
}

bool ccHObject::toFile_MeOnly(QFile& out) const
{
	assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));

	/*** ccHObject takes in charge the ccDrawableObject properties (which is not a ccSerializableObject) ***/

	//'visible' state (dataVersion>=20)
	if (out.write(reinterpret_cast<const char*>(&m_visible), sizeof(bool)) < 0)
		return WriteError();
	//'lockedVisibility' state (dataVersion>=20)
	if (out.write(reinterpret_cast<const char*>(&m_lockedVisibility), sizeof(bool)) < 0)
		return WriteError();
	//'colorsDisplayed' state (dataVersion>=20)
	if (out.write(reinterpret_cast<const char*>(&m_colorsDisplayed), sizeof(bool)) < 0)
		return WriteError();
	//'normalsDisplayed' state (dataVersion>=20)
	if (out.write(reinterpret_cast<const char*>(&m_normalsDisplayed), sizeof(bool)) < 0)
		return WriteError();
	//'sfDisplayed' state (dataVersion>=20)
	if (out.write(reinterpret_cast<const char*>(&m_sfDisplayed), sizeof(bool)) < 0)
		return WriteError();
	//'colorIsOverriden' state (dataVersion>=20)
	if (out.write(reinterpret_cast<const char*>(&m_colorIsOverriden), sizeof(bool)) < 0)
		return WriteError();
	if (m_colorIsOverriden)
	{
		//'tempColor' (dataVersion>=20)
		if (out.write(reinterpret_cast<const char*>(m_tempColor.rgb), sizeof(ColorCompType) * 3) < 0)
		{
			return WriteError();
		}
	}
	//'glTransEnabled' state (dataVersion>=20)
	if (out.write(reinterpret_cast<const char*>(&m_glTransEnabled), sizeof(bool)) < 0)
		return WriteError();
	if (m_glTransEnabled)
	{
		if (!m_glTrans.toFile(out))
		{
			return false;
		}
	}

	//'showNameIn3D' state (dataVersion>=24)
	if (out.write(reinterpret_cast<const char*>(&m_showNameIn3D), sizeof(bool)) < 0)
		return WriteError();

	return true;
}

bool ccHObject::fromFile_MeOnly(QFile& in, short dataVersion, int flags, LoadedIDMap& oldToNewIDMap)
{
	assert(in.isOpen() && (in.openMode() & QIODevice::ReadOnly));

	/*** ccHObject takes in charge the ccDrawableObject properties (which is not a ccSerializableObject) ***/

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
	if (in.read(reinterpret_cast<char*>(&m_colorIsOverriden), sizeof(bool)) < 0)
		return ReadError();
	if (m_colorIsOverriden)
	{
		//'tempColor' (dataVersion>=20)
		if (in.read(reinterpret_cast<char*>(m_tempColor.rgb), sizeof(ColorCompType) * 3) < 0)
			return ReadError();
	}
	//'glTransEnabled' state (dataVersion>=20)
	if (in.read(reinterpret_cast<char*>(&m_glTransEnabled), sizeof(bool)) < 0)
		return ReadError();
	if (m_glTransEnabled)
	{
        if (!m_glTrans.fromFile(in, dataVersion, flags, oldToNewIDMap))
		{
			return false;
		}
	}

	//'showNameIn3D' state (dataVersion>=24)
	if (dataVersion >= 24)
	{
		if (in.read(reinterpret_cast<char*>(&m_showNameIn3D), sizeof(bool)) < 0)
		{
			return WriteError();
		}
	}
	else
	{
		m_showNameIn3D = false;
	}

	return true;
}

void ccHObject::drawNameIn3D()
{
	QFont font = ecvDisplayTools::GetTextDisplayFont(); //takes rendering zoom into account!
	ecvDisplayTools::DisplayText(getName(),
		static_cast<int>(m_nameIn3DPos.x),
		static_cast<int>(m_nameIn3DPos.y),
		ecvDisplayTools::ALIGN_HMIDDLE | ecvDisplayTools::ALIGN_VMIDDLE,
		0.75f,
		nullptr,
		&font);
}

void ccHObject::draw(CC_DRAW_CONTEXT& context)
{
	// for polyline fast removement
	if (getRemoveFlag())
	{
		setRemoveType(context);
        context.removeViewID = getViewId();
		ecvDisplayTools::RemoveEntities(context);
		return;
	}

	//are we currently drawing objects in 2D or 3D?
	bool draw3D = MACRO_Draw3D(context);

	if (!isEnabled() &&
		!isKindOf(CV_TYPES::POINT_OCTREE) &&
		!isKindOf(CV_TYPES::POINT_KDTREE))
	{
		hideObject_recursive(true);
		// no need to do anything
		return;
	}

	//the entity must be either visible or selected, and of course it should be displayed in this context
	bool drawInThisContext = ((m_visible || m_selected));
	context.visible = m_visible;
	context.opacity = getOpacity();

	if (!isFixedId())
	{
        context.viewID = getViewId();
	}

	if (draw3D)
	{
		//apply 3D 'temporary' transformation (for display only)
		if (m_glTransEnabled)
		{
			//context.transformInfo.setRotMat(m_glTrans);
		}

		// LOD for clouds is enabled?
		if (context.decimateCloudOnMove
			&&	context.currentLODLevel > 0)
		{
			//only for real clouds
			drawInThisContext &= isA(CV_TYPES::POINT_CLOUD);
		}
	}

	// draw entity
	if (m_visible && drawInThisContext && context.forceRedraw)
	{
		if ((!m_selected || !MACRO_SkipSelected(context)) &&
			(m_selected || !MACRO_SkipUnselected(context)))
		{
			//enable clipping planes (if any)
			bool useClipPlanes = (draw3D && !m_clipPlanes.empty());
			if (useClipPlanes)
			{
				toggleClipPlanes(context, true);
			}

			drawMeOnly(context);

			//disable clipping planes (if any)
			if (useClipPlanes)
			{
				toggleClipPlanes(context, false);
			}
		}
	}

	// hide or show entities
	{
		setHideShowType(context);
		bool hasExist = ecvDisplayTools::HideShowEntities(context);
		if (!context.forceRedraw && m_forceRedraw && !hasExist)
		{
			if (!isA(CV_TYPES::OBJECT) && !isA(CV_TYPES::HIERARCHY_OBJECT) && !isA(CV_TYPES::TRANS_BUFFER))
			{
				setForceRedraw(false);
				CC_DRAW_CONTEXT newContext = context;
				newContext.forceRedraw = true;
				setRedrawFlagRecursive(true);
				draw(newContext);
			}
		}
	}

	//draw name - container objects are not visible but can still show a name
	if (m_showNameIn3D && !MACRO_DrawEntityNames(context))
	{
		if (MACRO_Draw3D(context))
		{
			//we have to comute the 2D position during the 3D pass!
			ccBBox bBox = getBB_recursive(true); //DGM: take the OpenGL features into account (as some entities are purely 'GL'!)
			if (bBox.isValid())
			{
				ccGLCameraParameters camera;
				ecvDisplayTools::GetGLCameraParameters(camera);

				CCVector3 C = bBox.getCenter();
				camera.project(C, m_nameIn3DPos);
			}
		}
		else if (MACRO_Draw2D(context) && MACRO_Foreground(context))
		{
			//then we can display the name during the 2D pass
			drawNameIn3D();
		}
	} 
	else
	{
		// label2d name have been managed by itself
		if (!isKindOf(CV_TYPES::LABEL_2D))
		{
			ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, getName()));
			ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_RECTANGLE_2D, getName()));
		}
	}

	//draw entity's children
	for (auto child : m_children)
	{
		child->draw(context);
	}

	//if the entity is currently selected, we draw its bounding-box
	if (m_selected && draw3D && drawInThisContext && !MACRO_DrawEntityNames(context) && context.currentLODLevel == 0)
	{
		CC_DRAW_CONTEXT tempContext = context;
		tempContext.meshRenderingMode = MESH_RENDERING_MODE::ECV_WIREFRAME_MODE;
        tempContext.viewID = getViewId();
		drawBB(tempContext, tempContext.bbDefaultCol);
        tempContext.viewID = getViewId();
		showBB(tempContext);
	} 
	
	if (!m_selected && draw3D)
	{
		CC_DRAW_CONTEXT context;
        context.viewID = getViewId();
		hideBB(context);
	}

	// reset redraw flag to true and forceRedraw flag to false(default)
	setRedraw(true);
	setForceRedraw(false);
}

void ccHObject::updateNameIn3DRecursive()
{
	if (nameShownIn3D())
	{
		ccBBox bBox = getBB_recursive(true); //DGM: take the OpenGL features into account (as some entities are purely 'GL'!)
		if (bBox.isValid())
		{
			ccGLCameraParameters camera;
			ecvDisplayTools::GetGLCameraParameters(camera);

			CCVector3 C = bBox.getCenter();
			camera.project(C, m_nameIn3DPos);

			// clear history name 3D
			ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, getName()));
			
			// draw name in 3D
			drawNameIn3D();
		}
	}

	for (auto child : m_children)
	{
		child->updateNameIn3DRecursive();
	}
}

void ccHObject::setHideShowType(CC_DRAW_CONTEXT & context)
{
	context.hideShowEntityType = ecvDisplayTools::ConvertToEntityType(getClassID());
}

ENTITY_TYPE ccHObject::getEntityType() const
{
	return ecvDisplayTools::ConvertToEntityType(getClassID());
}

void ccHObject::setRemoveType(CC_DRAW_CONTEXT & context)
{
	context.removeEntityType = ecvDisplayTools::ConvertToEntityType(getClassID());
}

void ccHObject::hideBB(CC_DRAW_CONTEXT context)
{
	context.hideShowEntityType = ENTITY_TYPE::ECV_SHAPE;
	context.viewID = QString("BBox-") + context.viewID;
	context.visible = false;
	ecvDisplayTools::HideShowEntities(context);
}

void ccHObject::showBB(CC_DRAW_CONTEXT context)
{
	context.hideShowEntityType = ENTITY_TYPE::ECV_SHAPE;
	context.viewID = QString("BBox-") + context.viewID;
	context.visible = true;
    ecvDisplayTools::HideShowEntities(context);
}

ccHObject::GlobalBoundingBox ccHObject::getOwnGlobalBB(bool withGLFeatures/*=false*/)
{
    //by default this method returns the local bounding-box!
    ccBBox box = getOwnBB(false);
    return GlobalBoundingBox(CCVector3d::fromArray(box.minCorner().u), CCVector3d::fromArray(box.maxCorner().u));
}

bool ccHObject::getOwnGlobalBB(CCVector3d& minCorner, CCVector3d& maxCorner)
{
    //by default this method returns the local bounding-box!
    ccBBox box = getOwnBB(false);
    minCorner = CCVector3d::fromArray(box.minCorner().u);
    maxCorner = CCVector3d::fromArray(box.maxCorner().u);
    return box.isValid();
}

ccHObject::GlobalBoundingBox ccHObject::getGlobalBB_recursive(bool withGLFeatures/*=false*/, bool onlyEnabledChildren/*=true*/)
{
    GlobalBoundingBox box = getOwnGlobalBB(withGLFeatures);

    for (auto child : m_children)
    {
        if (!onlyEnabledChildren || child->isEnabled())
        {
            box += child->getGlobalBB_recursive(withGLFeatures, onlyEnabledChildren);
        }
    }

    return box;
}


void ccHObject::hideObject_recursive(bool recursive)
{
	// hide obj recursively.
	std::vector<hideInfo> hdInfos;
	CC_DRAW_CONTEXT context;
	getTypeID_recursive(hdInfos, recursive);
	context.visible = false;
	for (const hideInfo& hdInfo : hdInfos)
	{
		if (hdInfo.hideType == ENTITY_TYPE::ECV_NONE) continue;

		context.hideShowEntityType = hdInfo.hideType;
        context.viewID = hdInfo.hideId;
        // hide obj bbox
        hideBB(context);
        context.viewID = hdInfo.hideId;

        ccHObject* obj = find(hdInfo.hideId.toUInt());

		if (hdInfo.hideType == ENTITY_TYPE::ECV_2DLABLE)
		{
			assert(obj && obj->isA(CV_TYPES::LABEL_2D));
			cc2DLabel* label2d = ccHObjectCaster::To2DLabel(obj);
			label2d->setEnabled(false);
			label2d->updateLabel();
			continue;
		}
		else if (hdInfo.hideType == ENTITY_TYPE::ECV_2DLABLE_VIEWPORT)
		{
			assert(obj && obj->isA(CV_TYPES::VIEWPORT_2D_LABEL));
			cc2DViewportLabel* label2d = ccHObjectCaster::To2DViewportLabel(obj);
			label2d->setEnabled(false);
			label2d->update2DLabelView(context, true);
			continue;
		}
        else if (hdInfo.hideType == ENTITY_TYPE::ECV_SENSOR)
        {
            ccSensor* sensor = ccHObjectCaster::ToSensor(obj);
            if (sensor)
            {
                sensor->hideShowDrawings(context);
                continue;
            }
        }
        else if (hdInfo.hideType == ENTITY_TYPE::ECV_MESH)
        {
            // try hide primitives
            ccGenericPrimitive* prim = ccHObjectCaster::ToPrimitive(obj);
            if (prim)
            {
                prim->hideShowDrawings(context);
                continue;
            }
        }

		context.viewID = hdInfo.hideId;
		ecvDisplayTools::HideShowEntities(context);
	}
}

void ccHObject::redrawDisplay(bool forceRedraw/* = true*/, bool only2D/* = false*/) {
	ecvDisplayTools::RedrawDisplay(only2D, forceRedraw);
}
