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

#include "ecvBBox.h"

// LOCAL
#include "ecvOrientedBBox.h"
#include "ecvDisplayTools.h"
#include <Logging.h>

using namespace cloudViewer;

ecvOrientedBBox ccBBox::getOrientedBoundingBox() const
{
	return ecvOrientedBBox::CreateFromAxisAlignedBoundingBox(*this);
}

ccBBox& ccBBox::transform(const Eigen::Matrix4d & transformation)
{
	utility::LogError(
		"A general transform of a ccBBox would not be axis "
		"aligned anymore, convert it to a OrientedBoundingBox first");
	return *this;
}

ccBBox& ccBBox::translate(const Eigen::Vector3d& translation, bool relative)
{
	if (relative) {
		m_bbMin += translation;
		m_bbMax += translation;
	}
	else {
		const Eigen::Vector3d half_extent = getHalfExtent();
		m_bbMin = CCVector3::fromArray(translation - half_extent);
		m_bbMax = CCVector3::fromArray(translation + half_extent);
	}
	return *this;
}

ccBBox& ccBBox::scale(const double s, const Eigen::Vector3d& center)
{
	m_bbMin = s * (m_bbMin - center) + center;
	m_bbMax = s * (m_bbMax - center) + center;
	return *this;
}

ccBBox& ccBBox::rotate(const Eigen::Matrix3d & R, const Eigen::Vector3d& center)
{
	utility::LogError(
		"A rotation of a ccBBox would not be axis aligned "
		"anymore, convert it to an ecvOrientedBBox first");
	return *this;
}

const ccBBox& ccBBox::operator+=(const ccBBox& other)
{
	if (isEmpty()) {
		this->m_bbMin = other.minCorner();
		this->m_bbMax = other.maxCorner();
		this->setValidity(true);
	}
	else if (!other.isEmpty()) {
		this->add(other.minCorner());
		this->add(other.maxCorner());
		this->setValidity(true);
	}
	return *this;
}

const ccBBox& ccBBox::operator+=(const CCVector3& aVector)
{
	if (m_valid)
	{
		m_bbMin += aVector;
		m_bbMax += aVector;
	}

	return *this;
}

void ccBBox::draw(CC_DRAW_CONTEXT& context, const ecvColor::Rgb& col) const
{
	if (!ecvDisplayTools::GetMainWindow())
	{
		return;
	}
	context.bbDefaultCol = col;
	context.viewID = QString("BBox-") + context.viewID;
	ecvDisplayTools::DrawBBox(context, this);
}

ccBBox ccBBox::CreateFromPoints(const std::vector<CCVector3>& points) {
	ccBBox box;
	if (points.empty()) {
		box.minCorner() = CCVector3(0.0f, 0.0f, 0.0f);
		box.maxCorner() = CCVector3(0.0f, 0.0f, 0.0f);
		box.setValidity(false);
	}
	else
	{
		for (auto &pt : points)
		{
			box.add(pt);
		}
	}
	box.setValidity(box.getMaxExtent() > 0);
	return box;
}

ccBBox ccBBox::CreateFromPoints(
	const std::vector<Eigen::Vector3d>& points) {
	ccBBox box;
	if (points.empty()) {
		box.minCorner() = CCVector3(0.0f, 0.0f, 0.0f);
		box.maxCorner() = CCVector3(0.0f, 0.0f, 0.0f);
	}
	else {
		box.minCorner() = std::accumulate(
			points.begin(), points.end(), points[0],
			[](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
			return a.array().min(b.array()).matrix();
		});
		box.maxCorner() = std::accumulate(
			points.begin(), points.end(), points[0],
			[](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
			return a.array().max(b.array()).matrix();
		});
	}

	box.setValidity(!box.isEmpty());
	return box;
}

std::vector<Eigen::Vector3d> ccBBox::getBoxPoints() const
{
	std::vector<Eigen::Vector3d> points(8);
	Eigen::Vector3d extent = getExtent();
	Eigen::Vector3d min_bound = CCVector3d::fromArray(m_bbMin);
	Eigen::Vector3d max_bound = CCVector3d::fromArray(m_bbMax);
	points[0] = min_bound;
	points[1] = min_bound + Eigen::Vector3d(extent(0), 0, 0);
	points[2] = min_bound + Eigen::Vector3d(0, extent(1), 0);
	points[3] = min_bound + Eigen::Vector3d(0, 0, extent(2));
	points[4] = max_bound;
	points[5] = max_bound - Eigen::Vector3d(extent(0), 0, 0);
	points[6] = max_bound - Eigen::Vector3d(0, extent(1), 0);
	points[7] = max_bound - Eigen::Vector3d(0, 0, extent(2));
	return points;
}

std::string ccBBox::getPrintInfo() const {
	return fmt::format("[({:.4f}, {:.4f}, {:.4f}) - ({:.4f}, {:.4f}, {:.4f})]",
		m_bbMin(0), m_bbMin(1), m_bbMin(2),
		m_bbMax(0), m_bbMax(1), m_bbMax(2));
}

const ccBBox ccBBox::operator * (const ccGLMatrix& mat)
{
	ccBBox rotatedBox;

	if (m_valid)
	{
		rotatedBox.add(mat * m_bbMin);
		rotatedBox.add(mat * CCVector3(m_bbMin.x,m_bbMin.y,m_bbMax.z));
		rotatedBox.add(mat * CCVector3(m_bbMin.x,m_bbMax.y,m_bbMin.z));
		rotatedBox.add(mat * CCVector3(m_bbMax.x,m_bbMin.y,m_bbMin.z));
		rotatedBox.add(mat * m_bbMax);
		rotatedBox.add(mat * CCVector3(m_bbMin.x,m_bbMax.y,m_bbMax.z));
		rotatedBox.add(mat * CCVector3(m_bbMax.x,m_bbMax.y,m_bbMin.z));
		rotatedBox.add(mat * CCVector3(m_bbMax.x,m_bbMin.y,m_bbMax.z));
	}

	return rotatedBox;
}

const ccBBox ccBBox::operator * (const ccGLMatrixd& mat)
{
	ccBBox rotatedBox;

	if (m_valid)
	{
		rotatedBox.add(mat * m_bbMin);
		rotatedBox.add(mat * CCVector3(m_bbMin.x,m_bbMin.y,m_bbMax.z));
		rotatedBox.add(mat * CCVector3(m_bbMin.x,m_bbMax.y,m_bbMin.z));
		rotatedBox.add(mat * CCVector3(m_bbMax.x,m_bbMin.y,m_bbMin.z));
		rotatedBox.add(mat * m_bbMax);
		rotatedBox.add(mat * CCVector3(m_bbMin.x,m_bbMax.y,m_bbMax.z));
		rotatedBox.add(mat * CCVector3(m_bbMax.x,m_bbMax.y,m_bbMin.z));
		rotatedBox.add(mat * CCVector3(m_bbMax.x,m_bbMin.y,m_bbMax.z));
	}

	return rotatedBox;
}
