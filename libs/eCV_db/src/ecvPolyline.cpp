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

#include "ecvPolyline.h"

//Local
#include "ecvPointCloud.h"
#include "ecvCone.h"
#include "ecvDisplayTools.h"

ccPolyline::ccPolyline(GenericIndexedCloudPersist* associatedCloud)
	: Polyline(associatedCloud)
	, ccShiftedObject("Polyline")
{
	set2DMode(false);
	setTransformFlag(true);
	setForeground(true);
	setVisible(true);
	lockVisibility(false);
	setColor(ecvColor::white);
	showVertices(false);
	setVertexMarkerWidth(3);
	setWidth(0);
	showArrow(false, 0, 0);

	ccGenericPointCloud* cloud = dynamic_cast<ccGenericPointCloud*>(associatedCloud);
	if (cloud)
	{
		setGlobalScale(cloud->getGlobalScale());
		setGlobalShift(cloud->getGlobalShift());
	}
}

ccPolyline::ccPolyline(ccPointCloud& associatedCloud)
	: Polyline(associatedCloud.cloneThis(nullptr))
	, ccShiftedObject("Polyline")
{
	this->set2DMode(false);
	this->setTransformFlag(true);
	this->setForeground(true);
	this->setVisible(true);
	this->lockVisibility(false);
	this->setColor(ecvColor::white);
	this->showColors(true);
	this->showVertices(false);
	this->setVertexMarkerWidth(3);
	this->setWidth(2);
	this->showArrow(false, 0, 0);

	int verticesCount = getAssociatedCloud()->size();
	if (this->reserve(verticesCount))
	{
		this->addPointIndex(0, verticesCount);
		this->setVisible(true);

		bool closed = false;
		CCVector3 start = CCVector3::fromArray(getAssociatedCloud()->getPoint(0)->u);
		CCVector3 end = CCVector3::fromArray(getAssociatedCloud()->getPoint(verticesCount - 1)->u);
        if (cloudViewer::LessThanEpsilon((end - start).norm()))
		{
			closed = true;
		} else {
			closed = false;
		}

		this->setClosed(closed);
		
		setGlobalScale(associatedCloud.getGlobalScale());
		setGlobalShift(associatedCloud.getGlobalShift());
	}
	else
	{
		cloudViewer::utility::LogError("[ccPolyline] not enough memory!");
	}
}

ccPolyline::ccPolyline(const ccPolyline& poly)
	: Polyline(0)
	, ccShiftedObject(poly)
{
	initWith(nullptr, poly);
}

bool ccPolyline::initWith(ccPointCloud* vertices, const ccPolyline& poly)
{
	bool success = true;
	if (!vertices)
	{
		ccPointCloud* cloud = dynamic_cast<ccPointCloud*>(poly.m_theAssociatedCloud);
		ccPointCloud* clone = cloud ? cloud->partialClone(&poly) : ccPointCloud::From(&poly);
		if (clone)
		{
			if (cloud)
				clone->setName(cloud->getName()); //as 'partialClone' adds the '.extract' suffix by default
			else
				clone->setGLTransformationHistory(poly.getGLTransformationHistory());
		}
		else
		{
			//not enough memory?
			CVLog::Warning("[ccPolyline::initWith] Not enough memory to duplicate vertices!");
			success = false;
		}

		vertices = clone;
	}

	if (vertices)
	{
		setAssociatedCloud(vertices);
		addChild(vertices);
		//vertices->setEnabled(false);
		assert(m_theAssociatedCloud);
		if (m_theAssociatedCloud)
		{
			if (!addPointIndex(0, m_theAssociatedCloud->size()))
			{
				CVLog::Warning("[ccPolyline::initWith] Not enough memory");
				success = false;
			}
		}
	}

	importParametersFrom(poly);

	return success;
}

void ccPolyline::importParametersFrom(const ccPolyline& poly)
{
	setClosed(poly.m_isClosed);
	set2DMode(poly.m_mode2D);
	setForeground(poly.m_foreground);
	setVisible(poly.isVisible());
	lockVisibility(poly.isVisiblityLocked());
	setColor(poly.m_rgbColor);
	setWidth(poly.m_width);
	showColors(poly.colorsShown());
	showVertices(poly.verticesShown());
	setVertexMarkerWidth(poly.getVertexMarkerWidth());
	setVisible(poly.isVisible());
	showArrow(m_showArrow,m_arrowIndex,m_arrowLength);
	setGlobalScale(poly.getGlobalScale());
	setGlobalShift(poly.getGlobalShift());
	setGLTransformationHistory(poly.getGLTransformationHistory());
	setMetaData(poly.metaData());
}

void ccPolyline::set2DMode(bool state)
{
	m_mode2D = state;
}

void ccPolyline::setForeground(bool state)
{
	m_foreground = state;
}

void ccPolyline::showArrow(bool state, unsigned vertIndex, PointCoordinateType length)
{
	m_showArrow = state;
	m_arrowIndex = vertIndex;
	m_arrowLength = length;
}

ccBBox ccPolyline::getOwnBB(bool withGLFeatures/*=false*/)
{
	ccBBox emptyBox;
	getBoundingBox(emptyBox.minCorner(), emptyBox.maxCorner());
	emptyBox.setValidity((!is2DMode() || !withGLFeatures) && size() != 0); //a 2D polyline is considered as a purely 'GL' fature
	return emptyBox;
}

bool ccPolyline::hasColors() const
{
	return true;
}

void ccPolyline::applyGLTransformation(const ccGLMatrix& trans)
{
	//transparent call
	ccHObject::applyGLTransformation(trans);

	//invalidate the bounding-box
	//(and we hope the vertices will be updated as well!)
	invalidateBoundingBox();
}

//unit arrow
static QSharedPointer<ccCone> c_unitArrow(0);

void ccPolyline::drawMeOnly(CC_DRAW_CONTEXT& context)
{
	unsigned vertCount = size();
	if (vertCount < 2)
		return;

	bool draw = false;

	if (MACRO_Draw3D(context))
	{
		draw = !m_mode2D;
	}
	else if (m_mode2D)
	{
		bool drawFG = MACRO_Foreground(context);
		draw = ((drawFG && m_foreground) || (!drawFG && !m_foreground));
	}

	if (!draw)
		return;

	if (ecvDisplayTools::GetMainWindow() == nullptr)
		return;

	if (isColorOverriden())
	{
		context.defaultPolylineColor = getTempColor();
	}
	else if (colorsShown())
	{
		context.defaultPolylineColor = m_rgbColor;
	}

	if (m_showVertices)
	{
		context.defaultPointSize = static_cast<unsigned>(m_vertMarkWidth);
	}

	//display polyline
	if (vertCount > 1)
	{
		if (m_width != 0)
		{
			context.currentLineWidth = m_width;
		} 
		else
		{
			context.currentLineWidth = context.defaultLineWidth;
		}

		ecvDisplayTools::Draw(context, this);

		//display arrow
		if (m_showArrow && m_arrowIndex < vertCount && (m_arrowIndex > 0 || m_isClosed))
		{
			const CCVector3* P0 = getPoint(m_arrowIndex == 0 ? vertCount - 1 : m_arrowIndex - 1);
			const CCVector3* P1 = getPoint(m_arrowIndex);
			//direction of the last polyline chunk
			CCVector3 u = *P1 - *P0;
			u.normalize();

			if (m_mode2D)
			{
				u *= -m_arrowLength;
				static const PointCoordinateType s_defaultArrowAngle = static_cast<PointCoordinateType>(15.0 * CV_DEG_TO_RAD);
				static const PointCoordinateType cost = cos(s_defaultArrowAngle);
				static const PointCoordinateType sint = sin(s_defaultArrowAngle);
				CCVector3 A(cost * u.x - sint * u.y, sint * u.x + cost * u.y, 0);
				CCVector3 B(cost * u.x + sint * u.y, -sint * u.x + cost * u.y, 0);
				//glFunc->glBegin(GL_POLYGON);
				//ccGL::Vertex3v(glFunc, (A + *P1).u);
				//ccGL::Vertex3v(glFunc, (B + *P1).u);
				//ccGL::Vertex3v(glFunc, (*P1).u);
				//glFunc->glEnd();
			}
			else
			{
				if (!c_unitArrow)
				{
					c_unitArrow = QSharedPointer<ccCone>(new ccCone(0.5, 0.0, 1.0));
					c_unitArrow->showColors(true);
					c_unitArrow->showNormals(false);
					c_unitArrow->setVisible(true);
					c_unitArrow->setEnabled(true);
				}
				if (colorsShown())
					c_unitArrow->setTempColor(m_rgbColor);
				else
					c_unitArrow->setTempColor(context.pointsDefaultCol);
				//build-up unit arrow own 'context'
				CC_DRAW_CONTEXT markerContext = context;
				markerContext.drawingFlags &= (~CC_DRAW_ENTITY_NAMES); // we must remove the 'push name flag' so that the sphere doesn't push its own!

				markerContext.transformInfo.setTranslationStart(CCVector3(P1->x, P1->y, P1->z));
				ccGLMatrixd rotMat = ccGLMatrixd::FromToRotation(CCVector3d(u.x, u.y, u.z), CCVector3d(0, 0, PC_ONE));
                markerContext.transformInfo.setTransformation(rotMat.inverse(), false);
				markerContext.transformInfo.setScale(CCVector3(m_arrowLength, m_arrowLength, m_arrowLength));
				markerContext.transformInfo.setTranslationEnd(CCVector3(0.0, 0.0, -0.5));
				c_unitArrow->draw(markerContext);
			}
		}
	}
}

void ccPolyline::setWidth(PointCoordinateType width)
{
	m_width = width;
}

bool ccPolyline::toFile_MeOnly(QFile& out) const
{
	if (!ccHObject::toFile_MeOnly(out))
		return false;

	//we can't save the associated cloud here (as it may be shared by multiple polylines)
	//so instead we save it's unique ID (dataVersion>=28)
	//WARNING: the cloud must be saved in the same BIN file! (responsibility of the caller)
	ccPointCloud* vertices = dynamic_cast<ccPointCloud*>(m_theAssociatedCloud);
	if (!vertices)
	{
		CVLog::Warning("[ccPolyline::toFile_MeOnly] Polyline vertices is not a ccPointCloud structure?!");
		return false;
	}
	uint32_t vertUniqueID = (m_theAssociatedCloud ? (uint32_t)vertices->getUniqueID() : 0);
	if (out.write((const char*)&vertUniqueID, 4) < 0)
		return WriteError();

	//number of points (references to) (dataVersion>=28)
	uint32_t pointCount = size();
	if (out.write((const char*)&pointCount, 4) < 0)
		return WriteError();

	//points (references to) (dataVersion>=28)
	for (uint32_t i = 0; i < pointCount; ++i)
	{
		uint32_t pointIndex = getPointGlobalIndex(i);
		if (out.write((const char*)&pointIndex, 4) < 0)
			return WriteError();
	}

	//'global shift & scale' (dataVersion>=39)
	saveShiftInfoToFile(out);

	QDataStream outStream(&out);

	//Closing state (dataVersion>=28)
	outStream << m_isClosed;

	//RGB Color (dataVersion>=28)
	outStream << m_rgbColor.r;
	outStream << m_rgbColor.g;
	outStream << m_rgbColor.b;

	//2D mode (dataVersion>=28)
	outStream << m_mode2D;

	//Foreground mode (dataVersion>=28)
	outStream << m_foreground;

	//The width of the line (dataVersion>=31)
	outStream << m_width;

	return true;
}

bool ccPolyline::fromFile_MeOnly(QFile& in, short dataVersion, int flags, LoadedIDMap& oldToNewIDMap)
{
    if (!ccHObject::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
		return false;

	if (dataVersion < 28)
		return false;

	//as the associated cloud (=vertices) can't be saved directly (as it may be shared by multiple polylines)
	//we only store its unique ID (dataVersion>=28) --> we hope we will find it at loading time (i.e. this
	//is the responsibility of the caller to make sure that all dependencies are saved together)
	uint32_t vertUniqueID = 0;
	if (in.read((char*)&vertUniqueID, 4) < 0)
		return ReadError();
	//[DIRTY] WARNING: temporarily, we set the vertices unique ID in the 'm_associatedCloud' pointer!!!
	*(uint32_t*)(&m_theAssociatedCloud) = vertUniqueID;

	//number of points (references to) (dataVersion>=28)
	uint32_t pointCount = 0;
	if (in.read((char*)&pointCount, 4) < 0)
		return ReadError();
	if (!reserve(pointCount))
		return false;

	//points (references to) (dataVersion>=28)
	for (uint32_t i = 0; i < pointCount; ++i)
	{
		uint32_t pointIndex = 0;
		if (in.read((char*)&pointIndex, 4) < 0)
			return ReadError();
		addPointIndex(pointIndex);
	}

	//'global shift & scale' (dataVersion>=39)
	if (dataVersion >= 39)
	{
		if (!loadShiftInfoFromFile(in))
			return ReadError();
	}
	else
	{
		m_globalScale = 1.0;
		m_globalShift = CCVector3d(0,0,0);
	}

	QDataStream inStream(&in);

	//Closing state (dataVersion>=28)
	inStream >> m_isClosed;

	//RGB Color (dataVersion>=28)
	inStream >> m_rgbColor.r;
	inStream >> m_rgbColor.g;
	inStream >> m_rgbColor.b;

	//2D mode (dataVersion>=28)
	inStream >> m_mode2D;

	//Foreground mode (dataVersion>=28)
	inStream >> m_foreground;

	//Width of the line (dataVersion>=31)
	if (dataVersion >= 31)
		ccSerializationHelper::CoordsFromDataStream(inStream,flags,&m_width,1);
	else
		m_width = 0;

	return true;
}

bool ccPolyline::split(	PointCoordinateType maxEdgeLength,
						std::vector<ccPolyline*>& parts)
{
	parts.clear();

	//not enough vertices?
	unsigned vertCount = size();
	if (vertCount <= 2)
	{
		parts.push_back(new ccPolyline(*this));
		return true;
	}

	unsigned startIndex = 0;
	unsigned lastIndex = vertCount-1;
	while (startIndex <= lastIndex)
	{
		unsigned stopIndex = startIndex;
		while (stopIndex < lastIndex && (*getPoint(stopIndex+1) - *getPoint(stopIndex)).norm() <= maxEdgeLength)
		{
			++stopIndex;
		}

		//number of vertices for the current part
		unsigned partSize = stopIndex-startIndex+1;

		//if the polyline is closed we have to look backward for the first segment!
		if (startIndex == 0)
		{
			if (isClosed())
			{
				unsigned realStartIndex = vertCount;
				while (realStartIndex > stopIndex && (*getPoint(realStartIndex-1) - *getPoint(realStartIndex % vertCount)).norm() <= maxEdgeLength)
				{
					--realStartIndex;
				}

				if (realStartIndex == stopIndex)
				{
					//whole loop
					parts.push_back(new ccPolyline(*this));
					return true;
				}
				else if (realStartIndex < vertCount)
				{
					partSize += (vertCount - realStartIndex);
					assert(realStartIndex != 0);
					lastIndex = realStartIndex-1;
					//warning: we shift the indexes!
					startIndex = realStartIndex; 
					stopIndex += vertCount;
				}
			}
			else if (partSize == vertCount)
			{
				//whole polyline
				parts.push_back(new ccPolyline(*this));
				return true;
			}
		}

		if (partSize > 1) //otherwise we skip that point
		{
			//create the corresponding part
			cloudViewer::ReferenceCloud ref(m_theAssociatedCloud);
			if (!ref.reserve(partSize))
			{
				CVLog::Error("[ccPolyline::split] Not enough memory!");
				return false;
			}

			for (unsigned i=startIndex; i<=stopIndex; ++i)
			{
				ref.addPointIndex(i % vertCount);
			}

			ccPointCloud* vertices = dynamic_cast<ccPointCloud*>(m_theAssociatedCloud);
			ccPointCloud* subset = vertices ? vertices->partialClone(&ref) : ccPointCloud::From(&ref);
			ccPolyline* part = new ccPolyline(subset);
			part->initWith(subset, *this);
			part->setClosed(false); //by definition!
			parts.push_back(part);
		}

		//forward
		startIndex = (stopIndex % vertCount) + 1;
	}

	return true;
}

bool ccPolyline::add(const ccPointCloud& cloud)
{
	if (cloud.isEmpty())
	{
		return false;
	}

	if (!this->getAssociatedCloud())
	{
		ccPointCloud* vertices = const_cast<ccPointCloud &>(cloud).cloneThis(nullptr);
		return initWith(vertices, *this);
	}

	ccPointCloud* oldCloud = static_cast<ccPointCloud*>(m_theAssociatedCloud);
	if (!oldCloud)
	{
		cloudViewer::utility::LogWarning("[ccPolyline::add] invalid associated cloud!");
		return false;
	}
	unsigned int newCount = cloud.size();
	unsigned int currentSize = oldCloud->size();
	if (!oldCloud->reserve(currentSize + newCount))
	{
		cloudViewer::utility::LogWarning("[ccPolyline] Not enough memory!");
		return false;
	}

	//copy new indexes (warning: no duplicate check!)
	for (unsigned i = 0; i < newCount; ++i)
	{
		oldCloud->addPoint(*cloud.getPoint(i));
	}
	addPointIndex(currentSize, currentSize + newCount);

	return true;
}

PointCoordinateType ccPolyline::computeLength() const
{
	PointCoordinateType length = 0;

	unsigned vertCount = size();
	if (vertCount > 1 && m_theAssociatedCloud)
	{
		unsigned lastVert = isClosed() ? vertCount : vertCount-1;
		for (unsigned i=0; i<lastVert; ++i)
		{
			CCVector3 A;
			getPoint(i, A);
			CCVector3 B;
			getPoint((i + 1) % vertCount, B);

			length += (B - A).norm();
		}
	}

	return length;
}

unsigned ccPolyline::getUniqueIDForDisplay() const
{
	if (m_parent && m_parent->getParent() && m_parent->getParent()->isA(CV_TYPES::FACET))
		return m_parent->getParent()->getUniqueID();
	else
		return getUniqueID();
}

unsigned ccPolyline::segmentCount() const
{
	unsigned count = size();
	if (count && !isClosed())
	{
		--count;
	}
	return count;
}

void ccPolyline::setGlobalShift(const CCVector3d& shift)
{
	ccShiftedObject::setGlobalShift(shift);

	ccPointCloud* pc = dynamic_cast<ccPointCloud*>(m_theAssociatedCloud);
	if (pc && pc->getParent() == this)
	{
		//auto transfer the global shift info to the vertices
		pc->setGlobalShift(shift);
	}
}

void ccPolyline::setGlobalScale(double scale)
{
	ccShiftedObject::setGlobalScale(scale);

	ccPointCloud* pc = dynamic_cast<ccPointCloud*>(m_theAssociatedCloud);
	if (pc && pc->getParent() == this)
	{
		//auto transfer the global scale info to the vertices
		pc->setGlobalScale(scale);
	}
}

ccPointCloud* ccPolyline::samplePoints(	bool densityBased,
                                        double samplingParameter,
                                        bool withRGB)
{
	if (samplingParameter <= 0 || size() < 2)
	{
		assert(false);
		return nullptr;
	}

	//we must compute the total length of the polyline
	double L = this->computeLength();

	unsigned pointCount = 0;
	if (densityBased)
	{
		pointCount = static_cast<unsigned>(ceil(L * samplingParameter));
	}
	else
	{
		pointCount = static_cast<unsigned>(samplingParameter);
	}

	if (pointCount == 0)
	{
		assert(false);
		return nullptr;
	}

	//convert to real point cloud
	ccPointCloud* cloud = new ccPointCloud(getName() + "." + QObject::tr("sampled"));
	if (!cloud->reserve(pointCount))
	{
		CVLog::Warning("[ccPolyline::samplePoints] Not enough memory");
		delete cloud;
		return nullptr;
	}

	double samplingStep = L / pointCount;
	double s = 0.0; //current sampled point curvilinear position
	unsigned indexA = 0; //index of the segment start vertex
	double sA = 0.0; //curvilinear pos of the segment start vertex

	for (unsigned i = 0; i < pointCount; )
	{
		unsigned indexB = ((indexA + 1) % size());
		const CCVector3& A = *getPoint(indexA);
		const CCVector3& B = *getPoint(indexB);
		CCVector3 AB = B - A;
		double lAB = AB.normd();

		double relativePos = s - sA;
		if (relativePos >= lAB)
		{
			//specific case: last point
			if (i + 1 == pointCount)
			{
				//assert(relativePos < lAB * 1.01); //it should only be a rounding issue in the worst case
				relativePos = lAB;
			}
			else //skip this segment
			{
				++indexA;
				sA += lAB;
				continue;
			}
		}

		//now for the interpolation work
		double alpha = relativePos / lAB;
		alpha = std::max(alpha, 0.0); //just in case
		alpha = std::min(alpha, 1.0);

		CCVector3 P = A + static_cast<PointCoordinateType>(alpha) * AB;
		cloud->addPoint(P);

		//proceed to the next point
		++i;
		s += samplingStep;
	}

	if (withRGB)
	{
		if (isColorOverriden())
		{
			//we use the default 'temporary' color
			cloud->setRGBColor(getTempColor());
		}
		else if (colorsShown())
		{
			//we use the default color
			cloud->setRGBColor(m_rgbColor);
		}
	}

	//import parameters from the source
	cloud->setGlobalShift(getGlobalShift());
	cloud->setGlobalScale(getGlobalScale());
	cloud->setGLTransformationHistory(getGLTransformationHistory());

	return cloud;
}

Eigen::Vector3d ccPolyline::getMinBound() const {
	return CCVector3d::fromArray(m_bbox.minCorner());
}

Eigen::Vector3d ccPolyline::getMaxBound() const {
	return CCVector3d::fromArray(m_bbox.maxCorner());
}

Eigen::Vector3d ccPolyline::getGeometryCenter() const {
	return CCVector3d::fromArray(m_bbox.getCenter());
}

ccBBox ccPolyline::getAxisAlignedBoundingBox() const {
	std::vector<CCVector3> points;
	for (unsigned index : m_theIndexes)
	{
		points.push_back(*m_theAssociatedCloud->getPoint(index));
	}
	return ccBBox::CreateFromPoints(points);
}

ecvOrientedBBox ccPolyline::getOrientedBoundingBox() const {
	if (!m_theAssociatedCloud)
	{
		return ecvOrientedBBox();
	}

	std::vector<CCVector3> points;
	for (unsigned index : m_theIndexes)
	{
		points.push_back(*m_theAssociatedCloud->getPoint(index));
	}
	return ecvOrientedBBox::CreateFromPoints(points);
}

ccPolyline &ccPolyline::transform(const Eigen::Matrix4d &transformation) {
	GenericIndexedCloudPersist* asCloud = getAssociatedCloud();
	if (!asCloud)
	{
		return *this;
	}
	ccPointCloud* cloud = static_cast<ccPointCloud*>(asCloud);
	if (cloud)
	{
		cloud->transform(transformation);
	}
	
	return *this;
}

ccPolyline &ccPolyline::translate(const Eigen::Vector3d &translation, bool relative) {
	GenericIndexedCloudPersist* asCloud = getAssociatedCloud();
	if (!asCloud)
	{
		return *this;
	}
	ccPointCloud* cloud = static_cast<ccPointCloud*>(asCloud);
	if (cloud)
	{
		cloud->translate(translation, relative);
	}
	return *this;
}

ccPolyline &ccPolyline::scale(const double s, const Eigen::Vector3d &center) {
	GenericIndexedCloudPersist* asCloud = getAssociatedCloud();
	if (!asCloud)
	{
		return *this;
	}
	ccPointCloud* cloud = static_cast<ccPointCloud*>(asCloud);
	if (cloud)
	{
		cloud->scale(s, center);
	}
	return *this;
}

ccPolyline &ccPolyline::rotate(const Eigen::Matrix3d &R, const Eigen::Vector3d &center) {
	GenericIndexedCloudPersist* asCloud = getAssociatedCloud();
	if (!asCloud)
	{
		return *this;
	}
	ccPointCloud* cloud = static_cast<ccPointCloud*>(asCloud);
	if (cloud)
	{
		cloud->rotate(R, center);
	}
	return *this;
}

ccPolyline &ccPolyline::operator+=(const ccPolyline &polyline) {
	if (polyline.isEmpty()) return (*this);
	if (!polyline.getAssociatedCloud())
	{
		cloudViewer::utility::LogError("[ccPolyline] Cannot find associated cloud in polyline!");
		return (*this);
	}

	if (m_theAssociatedCloud == polyline.getAssociatedCloud())
	{
		if (!cloudViewer::ReferenceCloud::add(polyline))
		{
			cloudViewer::utility::LogError("[ccPolyline] Not enough memory!");
			return (*this);
		}
	}
	else
	{
		const ccPointCloud* cloud = static_cast<const ccPointCloud*>(
			polyline.getAssociatedCloud());
		if (!cloud || !add(*cloud))
		{
			cloudViewer::utility::LogWarning("[ccPolyline] adding ccPolyline failed!");
		}
	}

	return (*this);
}

ccPolyline &ccPolyline::operator=(const ccPolyline &polyline) {
	if (this == &polyline)
	{
		return (*this);
	}

	this->clear();
	*this += polyline;
	//import other parameters
	this->importParametersFrom(polyline);
	return (*this);
}

ccPolyline ccPolyline::operator+(const ccPolyline &polyline) const {
	return (ccPolyline(*this) += polyline);
}
