//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER BACKEND : qPCL                       #
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
//#                         COPYRIGHT: DAHAI LU                         #
//#                                                                        #
//##########################################################################
//
#include "cc2sm.h"

//Local
#include "my_point_types.h"
#include "PCLDisplayTools.h"
#include "PCLConv.h"

// PCL
#include <pcl/common/io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/impl/vtk_lib_io.hpp>

// VTK
#include <vtkPolyData.h>
#include <vtkFloatArray.h>

// CV_CORE_LIB
#include <CVTools.h>

// ECV_DB_LIB
#include <ecvAdvancedTypes.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvPolyline.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

//system
#include <assert.h>

using namespace pcl;

cc2smReader::cc2smReader(bool showMode/* = false*/):
	m_cc_cloud(nullptr),
	m_showMode(showMode),
	m_partialVisibility(false),
	m_visibilityNum(0)
{
}

cc2smReader::cc2smReader(const ccPointCloud * cc_cloud, bool showMode/* = false*/) :
	m_cc_cloud(cc_cloud),
	m_showMode(showMode),
	m_partialVisibility(false),
	m_visibilityNum(0)
{
	assert(m_cc_cloud);
	//count the number of visible points
	if (m_cc_cloud->isVisibilityTableInstantiated())
	{
		m_visibilityNum = 0;
		assert(m_cc_cloud->getTheVisibilityArray().size() == m_cc_cloud->size());
		m_partialVisibility = true;
		unsigned count = m_cc_cloud->size();
		for (unsigned i = 0; i < count; ++i)
		{
			if (m_cc_cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
			{
				++m_visibilityNum;
			}
		}
	}
}

PCLCloud::Ptr cc2smReader::getGenericField(std::string field_name) const
{
	PCLCloud::Ptr sm_cloud;

	if (field_name == "x")
	{
		sm_cloud = getOneOf(COORD_X);
	}
	else if (field_name == "y")
	{
		sm_cloud = getOneOf(COORD_Y);
	}
	else if (field_name == "z")
	{
		sm_cloud = getOneOf(COORD_Y);
	}
	else if (field_name == "normal_x")
	{
		sm_cloud = getOneOf(NORM_X);
	}
	else if (field_name == "normal_y")
	{
		sm_cloud = getOneOf(NORM_Y);
	}
	else if (field_name == "normal_z")
	{
		sm_cloud = getOneOf(NORM_Z);
	}
	else if (field_name == "xyz")
	{
		sm_cloud = getXYZ();
	}
	else if (field_name == "normal_xyz")
	{
		sm_cloud = getNormals();
	}
	else if (field_name == "rgb")
	{
		sm_cloud = getColors();
	}
	else //try to load the field from the scalar fields
	{
		sm_cloud = getFloatScalarField(field_name);
	}

	return sm_cloud;
}

unsigned cc2smReader::getvisibilityNum() const
{
	return m_partialVisibility ? m_visibilityNum : m_cc_cloud->size();
}

PCLCloud::Ptr cc2smReader::getOneOf(Fields field) const
{
	assert(m_cc_cloud);

	PCLCloud::Ptr sm_cloud;

	std::string name;
	unsigned char dim = 0;
	switch (field)
	{
	case COORD_X:
		name = "x";
		dim = 0;
		break;
	case COORD_Y:
		name = "y";
		dim = 1;
		break;
	case COORD_Z:
		name = "z";
		dim = 2;
		break;
	case NORM_X:
		if (!m_cc_cloud->hasNormals())
			return sm_cloud;
		name = "normal_x";
		dim = 0;
		break;
	case NORM_Y:
		if (!m_cc_cloud->hasNormals())
			return sm_cloud;
		name = "normal_y";
		dim = 1;
		break;
	case NORM_Z:
		if (!m_cc_cloud->hasNormals())
			return sm_cloud;
		name = "normal_z";
		dim = 2;
		break;
	default:
		//unhandled field?!
		assert(false);
		return sm_cloud;
	};

	assert(/*dim >= 0 && */dim <= 2);

	try
	{
		PointCloud<FloatScalar>::Ptr pcl_cloud (new PointCloud<FloatScalar>);

		unsigned pointCount = m_cc_cloud->size();
		unsigned realNum = m_partialVisibility ? m_visibilityNum : m_cc_cloud->size();
		pcl_cloud->resize(realNum);
		unsigned index = 0;
		for (unsigned i = 0; i < pointCount; ++i)
		{
			switch(field)
			{
			case COORD_X:
			case COORD_Y:
			case COORD_Z:
				{
					if (m_partialVisibility)
					{
						if (m_cc_cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
						{
							const CCVector3* P = m_cc_cloud->getPoint(i);
							pcl_cloud->at(index).S5c4laR = static_cast<float>(P->u[dim]);
							++index;
						}
					}
					else
					{
						const CCVector3* P = m_cc_cloud->getPoint(i);
						pcl_cloud->at(i).S5c4laR = static_cast<float>(P->u[dim]);
					}
				}
				break;
			case NORM_X:
			case NORM_Y:
			case NORM_Z:
				{
				if (m_partialVisibility)
				{
					if (m_cc_cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
					{
						const CCVector3& N = m_cc_cloud->getPointNormal(i);
						pcl_cloud->at(i).S5c4laR = static_cast<float>(N.u[dim]);
						++index;
					}
				}
				else
				{
					const CCVector3& N = m_cc_cloud->getPointNormal(i);
					pcl_cloud->at(i).S5c4laR = static_cast<float>(N.u[dim]);
				}
				}
				break;
			default:
				//unhandled field?!
				assert(false);
				break;
			};
		}

		sm_cloud = PCLCloud::Ptr(new PCLCloud);
		TO_PCL_CLOUD(*pcl_cloud, *sm_cloud);
		sm_cloud->fields[0].name = name;
	}
	catch(...)
	{
		//any error (memory, etc.)
		sm_cloud.reset();
	}
	return sm_cloud;
}

PointCloud<PointXYZ>::Ptr cc2smReader::getXYZ2() const
{
	assert(m_cc_cloud);

	PointCloud<PointXYZ>::Ptr pcl_cloud (new PointCloud<PointXYZ>);
	try
	{
		unsigned pointCount = m_cc_cloud->size();
		unsigned realNum = m_partialVisibility ? m_visibilityNum : m_cc_cloud->size();
		pcl_cloud->resize(realNum);
		unsigned index = 0;

		for (unsigned i = 0; i < pointCount; ++i)
		{
			if (m_partialVisibility)
			{
				if (m_cc_cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
				{
					const CCVector3* P = m_cc_cloud->getPoint(i);
					pcl_cloud->at(index).x = static_cast<float>(P->x);
					pcl_cloud->at(index).y = static_cast<float>(P->y);
					pcl_cloud->at(index).z = static_cast<float>(P->z);
					++index;
				}
			} 
			else
			{
				const CCVector3* P = m_cc_cloud->getPoint(i);
				pcl_cloud->at(i).x = static_cast<float>(P->x);
				pcl_cloud->at(i).y = static_cast<float>(P->y);
				pcl_cloud->at(i).z = static_cast<float>(P->z);
			}
		}
	}
	catch(...)
	{
		//any error (memory, etc.)
		pcl_cloud.reset();
	}

	return pcl_cloud;
}

PCLCloud::Ptr cc2smReader::getXYZ() const
{
	PCLCloud::Ptr sm_cloud;
	
	PointCloud<PointXYZ>::Ptr pcl_cloud = getXYZ2();
	if (pcl_cloud)
	{
		sm_cloud = PCLCloud::Ptr(new PCLCloud);
		TO_PCL_CLOUD(*pcl_cloud, *sm_cloud);
	}

	return sm_cloud;
}

PCLCloud::Ptr cc2smReader::getNormals() const
{
	if (!m_cc_cloud || !m_cc_cloud->hasNormals())
		return PCLCloud::Ptr(static_cast<PCLCloud*>(0));

	PCLCloud::Ptr sm_cloud (new PCLCloud);
	try
	{
		PointCloud<OnlyNormals>::Ptr pcl_cloud (new PointCloud<OnlyNormals>);

		unsigned pointCount = m_cc_cloud->size();
		unsigned realNum = m_partialVisibility ? m_visibilityNum : m_cc_cloud->size();
		pcl_cloud->resize(realNum);

		unsigned index = 0;
		for (unsigned i = 0; i < pointCount; ++i)
		{
			if (m_partialVisibility)
			{
				if (m_cc_cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
				{
					const CCVector3& N = m_cc_cloud->getPointNormal(i);
					pcl_cloud->at(index).normal_x = N.x;
					pcl_cloud->at(index).normal_y = N.y;
					pcl_cloud->at(index).normal_z = N.z;
					++index;
				}
			}
			else
			{
				const CCVector3& N = m_cc_cloud->getPointNormal(i);
				pcl_cloud->at(i).normal_x = N.x;
				pcl_cloud->at(i).normal_y = N.y;
				pcl_cloud->at(i).normal_z = N.z;
			}
		}

		TO_PCL_CLOUD(*pcl_cloud, *sm_cloud);
	}
	catch(...)
	{
		//any error (memory, etc.)
		sm_cloud.reset();
	}
	
	return sm_cloud;
}

PCLCloud::Ptr cc2smReader::getPointNormals() const
{
	PCLCloud::Ptr normals = getNormals();
	if (!normals)
	{
		return normals;
	}
	PCLCloud::Ptr xyzCloud = getXYZ();
	PCLCloud::Ptr sm_tmp(new PCLCloud); //temporary cloud
	if (xyzCloud)
	{
		pcl::concatenateFields(*normals, *xyzCloud, *sm_tmp);
	}
	return sm_tmp;
}

PCLCloud::Ptr cc2smReader::getColors() const
{
	if (!m_cc_cloud || !m_cc_cloud->hasColors())
		return PCLCloud::Ptr(static_cast<PCLCloud*>(0));

	PCLCloud::Ptr sm_cloud (new PCLCloud);
	try
	{
		PointCloud<OnlyRGB>::Ptr pcl_cloud (new PointCloud<OnlyRGB>);

		unsigned pointCount = m_cc_cloud->size();
		unsigned realNum = m_partialVisibility ? m_visibilityNum : m_cc_cloud->size();
		pcl_cloud->resize(realNum);
		unsigned index = 0;

		for (unsigned i = 0; i < pointCount; ++i)
		{
			if (m_partialVisibility)
			{
				if (m_cc_cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
				{
					const ecvColor::Rgb& rgb = m_cc_cloud->getPointColor(i);
					pcl_cloud->at(index).r = static_cast<uint8_t>(rgb.r);
					pcl_cloud->at(index).g = static_cast<uint8_t>(rgb.g);
					pcl_cloud->at(index).b = static_cast<uint8_t>(rgb.b);
					++index;
				}
			}
			else
			{
				const ecvColor::Rgb& rgb = m_cc_cloud->getPointColor(i);
				pcl_cloud->at(i).r = static_cast<uint8_t>(rgb.r);
				pcl_cloud->at(i).g = static_cast<uint8_t>(rgb.g);
				pcl_cloud->at(i).b = static_cast<uint8_t>(rgb.b);
			}
		}

		TO_PCL_CLOUD(*pcl_cloud, *sm_cloud);
	}
	catch(...)
	{
		//any error (memory, etc.)
		sm_cloud.reset();
	}

	return sm_cloud;
}

/** \brief Obtain the actual color for the input dataset as vtk scalars.
  * \param[out] scalars the output scalars containing the color for the dataset
  * \return true if the operation was successful (the handler is capable and
  * the input cloud was given as a valid pointer), false otherwise
  */
bool cc2smReader::getvtkScalars(vtkSmartPointer<vtkDataArray> &scalars, bool sfColors) const
{
	if (!m_cc_cloud) return false;

	if (!scalars)
		scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
	scalars->SetNumberOfComponents(3);
	unsigned nr_points = m_partialVisibility ? m_visibilityNum : m_cc_cloud->size();
	reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetNumberOfTuples(static_cast<vtkIdType>(nr_points));
	unsigned char* colors = reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->GetPointer(0);
	int j = 0;
	if (m_cc_cloud->hasScalarFields() && sfColors)
	{
		int sfIdx = m_cc_cloud->getCurrentDisplayedScalarFieldIndex();
		if (sfIdx < 0) return false;
		CVLib::ScalarField* scalar_field = m_cc_cloud->getScalarField(sfIdx);
		if (!scalar_field) return false;
		for (unsigned cp = 0; cp < nr_points; ++cp)
		{
			const ecvColor::Rgb* rgb =
				m_cc_cloud->getScalarValueColor(scalar_field->getValue(cp));

			if (m_partialVisibility)
			{
				if (m_cc_cloud->getTheVisibilityArray().at(cp) != POINT_VISIBLE)
					continue;

				colors[j] = static_cast<uint8_t>(rgb->r);
				colors[j + 1] = static_cast<uint8_t>(rgb->g);
				colors[j + 2] = static_cast<uint8_t>(rgb->b);
				j += 3;
			}
			else
			{
				int idx = static_cast<int> (cp) * 3;
				colors[idx] = static_cast<uint8_t>(rgb->r);
				colors[idx + 1] = static_cast<uint8_t>(rgb->g);
				colors[idx + 2] = static_cast<uint8_t>(rgb->b);
			}
		}

	}
	else if (m_cc_cloud->hasColors())
	{
		for (unsigned cp = 0; cp < nr_points; ++cp)
		{
			// Color every point
			const ecvColor::Rgb& rgb = m_cc_cloud->getPointColor(cp);
			if (m_partialVisibility)
			{
				if (m_cc_cloud->getTheVisibilityArray().at(cp) != POINT_VISIBLE)
					continue;

				colors[j] = static_cast<uint8_t>(rgb.r);
				colors[j + 1] = static_cast<uint8_t>(rgb.g);
				colors[j + 2] = static_cast<uint8_t>(rgb.b);
				j += 3;
			
			}
			else
			{
				int idx = static_cast<int> (cp) * 3;
				colors[idx] = static_cast<uint8_t>(rgb.r);
				colors[idx + 1] = static_cast<uint8_t>(rgb.g);
				colors[idx + 2] = static_cast<uint8_t>(rgb.b);
			}
		}
	}
	else
	{
		return false;
	}

	return true;
}

std::string cc2smReader::GetSimplifiedSFName(const std::string& ccSfName)
{
	QString simplified = QString::fromStdString(ccSfName).simplified();
	simplified.replace(' ', '_');
	return simplified.toStdString();
}

PCLCloud::Ptr cc2smReader::getFloatScalarField(const std::string& field_name) const
{
	assert(m_cc_cloud);

	int sfIdx = m_cc_cloud->getScalarFieldIndexByName(field_name.c_str());
	if (sfIdx < 0)
		return PCLCloud::Ptr(static_cast<PCLCloud*>(0));
	CVLib::ScalarField* scalar_field = m_cc_cloud->getScalarField(sfIdx);
	assert(scalar_field);

	PCLCloud::Ptr sm_cloud (new PCLCloud);
	try
	{
		if (m_showMode)
		{
			// only convert first scalar field to rgb color data
			if (m_cc_cloud->sfShown() && 
				sfIdx == m_cc_cloud->getCurrentDisplayedScalarFieldIndex())
			{
				PointCloud<OnlyRGB>::Ptr pcl_cloud(new PointCloud<OnlyRGB>);

				unsigned pointCount = m_cc_cloud->size();
				unsigned realNum = m_partialVisibility ? m_visibilityNum : m_cc_cloud->size();
				pcl_cloud->resize(realNum);
				unsigned index = 0;

				for (unsigned i = 0; i < pointCount; ++i)
				{
					if (m_partialVisibility)
					{
						if (m_cc_cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
						{
							ScalarType scalar = scalar_field->getValue(i);
							//pcl_cloud->at(index).S5c4laR = static_cast<float>(scalar);
							const ecvColor::Rgb* col = m_cc_cloud->getScalarValueColor(scalar);
							pcl_cloud->at(index).r = static_cast<uint8_t>(col->r);
							pcl_cloud->at(index).g = static_cast<uint8_t>(col->g);
							pcl_cloud->at(index).b = static_cast<uint8_t>(col->b);
							++index;
						}
					}
					else
					{
						ScalarType scalar = scalar_field->getValue(i);
						//pcl_cloud->at(i).S5c4laR = static_cast<float>(scalar);
						const ecvColor::Rgb* col = m_cc_cloud->getScalarValueColor(scalar);
						pcl_cloud->at(i).r = static_cast<uint8_t>(col->r);
						pcl_cloud->at(i).g = static_cast<uint8_t>(col->g);
						pcl_cloud->at(i).b = static_cast<uint8_t>(col->b);
					}
				}

				TO_PCL_CLOUD(*pcl_cloud, *sm_cloud);
			}
			else
			{
				sm_cloud = nullptr;
			}
		}
		else
		{
			PointCloud<FloatScalar>::Ptr pcl_cloud (new PointCloud<FloatScalar>);
			unsigned pointCount = m_cc_cloud->size();
			unsigned realNum = m_partialVisibility ? m_visibilityNum : m_cc_cloud->size();
			pcl_cloud->resize(realNum);
			unsigned index = 0;

			for (unsigned i = 0; i < pointCount; ++i)
			{
				if (m_partialVisibility)
				{
					if (m_cc_cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
					{
						ScalarType scalar = scalar_field->getValue(i);
						pcl_cloud->at(index).S5c4laR = static_cast<float>(scalar);
						++index;
					}
				}
				else
				{
					ScalarType scalar = scalar_field->getValue(i);
					pcl_cloud->at(i).S5c4laR = static_cast<float>(scalar);
				}
			}

			TO_PCL_CLOUD(*pcl_cloud, *sm_cloud);

			//Now change the name of the scalar field -> we cannot have any space into the field name
			//NOTE this is a little trick to put any number of scalar fields in a message PointCloud2 object
			//We use a point type with a generic scalar field named scalar. we load the scalar field and
			//then we change the name to the needed one
			sm_cloud->fields[0].name = GetSimplifiedSFName(field_name);
		}
	}
	catch(...)
	{
		//any error (memory, etc.)
		sm_cloud.reset();
	}

	return sm_cloud;
}

bool cc2smReader::checkIfFieldExists(const std::string& field_name) const
{
	if ( (field_name == "x") || (field_name == "y") || (field_name == "z") || (field_name == "xyz") )
		return (m_cc_cloud->size() != 0);

	else if ( (field_name == "normal_x") || (field_name == "normal_y") || (field_name == "normal_z") || (field_name == "normal_xyz") )
		return m_cc_cloud->hasNormals();

	else if (field_name == "rgb")
		return m_cc_cloud->hasColors();

	else
		return (m_cc_cloud->getScalarFieldIndexByName(field_name.c_str()) >= 0);
}

PCLCloud::Ptr cc2smReader::getAsSM(std::list<std::string>& requested_fields) const
{
	//preliminary check
	{
		for (std::list<std::string>::const_iterator it = requested_fields.begin(); it != requested_fields.end(); ++it)
		{
			bool exists = checkIfFieldExists(*it);
			if (!exists) //all check results must be true
				return PCLCloud::Ptr(static_cast<PCLCloud*>(0));
		}
	}

	//are we asking for x, y, and z all togheters?
	bool got_xyz = (std::find(requested_fields.begin(), requested_fields.end(), "xyz") != requested_fields.end());
	if (got_xyz)
	{
		//remove from the requested fields lists x y and z as single occurrencies
		requested_fields.erase(std::remove(requested_fields.begin(), requested_fields.end(), std::string("x")), requested_fields.end());
		requested_fields.erase(std::remove(requested_fields.begin(), requested_fields.end(), std::string("y")), requested_fields.end());
		requested_fields.erase(std::remove(requested_fields.begin(), requested_fields.end(), std::string("z")), requested_fields.end());
	}

	//same for normals
	bool got_normal_xyz = (std::find(requested_fields.begin(), requested_fields.end(), "normal_xyz") != requested_fields.end());
	if (got_normal_xyz)
	{
		requested_fields.erase(std::remove(requested_fields.begin(), requested_fields.end(), std::string("normal_x")), requested_fields.end());
		requested_fields.erase(std::remove(requested_fields.begin(), requested_fields.end(), std::string("normal_y")), requested_fields.end());
		requested_fields.erase(std::remove(requested_fields.begin(), requested_fields.end(), std::string("normal_z")), requested_fields.end());
	}

	//a vector for PointCloud2 clouds
	PCLCloud::Ptr firstCloud;

	//load and merge fields/clouds one-by-one
	{
		for (std::list<std::string>::const_iterator it = requested_fields.begin(); it != requested_fields.end(); ++it)
		{
			if (!firstCloud)
			{
				firstCloud = getGenericField(*it);
			}
			else
			{
				PCLCloud::Ptr otherCloud = getGenericField(*it);
				if (otherCloud)
				{
					PCLCloud::Ptr sm_tmp (new PCLCloud); //temporary cloud
					pcl::concatenateFields(*firstCloud, *otherCloud, *sm_tmp);
					firstCloud = sm_tmp;
				}
			}
		}
	}

	return firstCloud;
}

PCLCloud::Ptr cc2smReader::getAsSM(bool ignoreScalars) const
{
	//does the cloud have some points?
	if (!m_cc_cloud || m_cc_cloud->size() == 0)
	{
		assert(false);
		return PCLCloud::Ptr(static_cast<PCLCloud*>(0));
	}

	//container
	std::list<std::string> fields;
	try
	{
		fields.push_back("xyz");
		if (m_cc_cloud->hasNormals())
			fields.push_back("normal_xyz");

		if (m_cc_cloud->hasColors())
			fields.push_back("rgb");

		if (!ignoreScalars)
		{
			for (unsigned i = 0; i < m_cc_cloud->getNumberOfScalarFields(); ++i)
				fields.push_back(m_cc_cloud->getScalarField(static_cast<int>(i))->getName());
		}
	}
	catch (const std::bad_alloc&)
	{
		//not enough memory
		return PCLCloud::Ptr(static_cast<PCLCloud*>(0));
	}

	return getAsSM(fields);
}

PCLCloud::Ptr cc2smReader::getVtkPolyDataAsSM(vtkPolyData* const polydata) const
{
	// Set the colors of the pcl::PointCloud (if the pcl::PointCloud supports colors and the input vtkPolyData has colors)
	vtkUnsignedCharArray* colors = vtkUnsignedCharArray::SafeDownCast(polydata->GetPointData()->GetScalars());
	
	// Set the normals of the pcl::PointCloud (if the pcl::PointCloud supports normals and the input vtkPolyData has normals)
	vtkFloatArray* normals = vtkFloatArray::SafeDownCast(polydata->GetPointData()->GetNormals());
	PCLCloud::Ptr smCloud(new PCLCloud);
	if (colors && normals)
	{
		PointCloudRGBNormal::Ptr cloud(new PointCloudRGBNormal);
		pcl::io::vtkPolyDataToPointCloud(polydata, *cloud);
		TO_PCL_CLOUD(*cloud, *smCloud);
	}
	else if (colors && !normals)
	{
		PointCloudRGB::Ptr cloud(new PointCloudRGB);
		pcl::io::vtkPolyDataToPointCloud(polydata, *cloud);
		TO_PCL_CLOUD(*cloud, *smCloud);
	}
	else if (!colors && normals)
	{
		PointCloudNormal::Ptr cloud(new PointCloudNormal);
		pcl::io::vtkPolyDataToPointCloud(polydata, *cloud);
		TO_PCL_CLOUD(*cloud, *smCloud);
	}
	else if (!colors && !normals)
	{
		PointCloudT::Ptr cloud(new PointCloudT);
		pcl::io::vtkPolyDataToPointCloud(polydata, *cloud);
		TO_PCL_CLOUD(*cloud, *smCloud);
	}

	return smCloud;
}

PCLMesh::Ptr cc2smReader::getVtkPolyDataAsPclMesh(vtkPolyData * const polydata) const
{
	PCLMesh::Ptr plcMesh(new PCLMesh);
	vtkCellArray * mesh_polygons = polydata->GetPolys();
	if (!mesh_polygons || mesh_polygons->GetNumberOfCells() == 0)
	{
		return nullptr;
	}

	pcl::io::vtk2mesh(polydata, *plcMesh);
	return plcMesh;
}

PCLMesh::Ptr cc2smReader::getPclMesh(ccGenericMesh* mesh) {
	if (!mesh) return nullptr;

	const ccGenericPointCloud::VisibilityTableType& verticesVisibility = mesh->getAssociatedCloud()->getTheVisibilityArray();
        bool visFiltering = (verticesVisibility.size() >= mesh->getAssociatedCloud()->size());
	PCLMesh::Ptr pclMesh(new PCLMesh);

    if (!getPclCloud2(mesh, pclMesh->cloud)) {
        CVLog::Warning("[cc2smReader::getPclMesh] Failed to get pcl::PCLPointCloud2!");
        return nullptr;
    }

	//vertices visibility
	unsigned triNum = mesh->size();

	for (unsigned n = 0; n < triNum; ++n)
	{
		const CVLib::VerticesIndexes* tsi = mesh->getTriangleVertIndexes(n);
		if (visFiltering)
		{
			//we skip the triangle if at least one vertex is hidden
			if ((verticesVisibility[tsi->i1] != POINT_VISIBLE) ||
				(verticesVisibility[tsi->i2] != POINT_VISIBLE) ||
				(verticesVisibility[tsi->i3] != POINT_VISIBLE))
				continue;
		}

        pcl::Vertices tri;
        tri.vertices.push_back(n * 3 + 0);
        tri.vertices.push_back(n * 3 + 1);
        tri.vertices.push_back(n * 3 + 2);
		pclMesh->polygons.push_back(tri);
	}

	return pclMesh;
}

void getMaterial(ccMaterial::CShared inMaterial, PCLMaterial& outMaterial)
{
	assert(inMaterial);
    inMaterial->getTexture();
	std::string texFile = CVTools::FromQString(inMaterial->getTextureFilename());
	std::string texName = CVTools::FromQString(inMaterial->getName());
	// FIX special symbols bugs in vtk rendering system!
    texName = CVTools::ExtractDigitAlpha(texName);
	const ecvColor::Rgbaf& ambientColor = inMaterial->getAmbient();
	const ecvColor::Rgbaf& diffuseColor = inMaterial->getDiffuseFront();
	const ecvColor::Rgbaf& specularColor = inMaterial->getSpecular();
	float shininess = inMaterial->getShininessFront();

	outMaterial.tex_name = texName;
	outMaterial.tex_file = texFile;
	outMaterial.tex_Ka.r = ambientColor.r;
	outMaterial.tex_Ka.g = ambientColor.g;
	outMaterial.tex_Ka.b = ambientColor.b;
	outMaterial.tex_Kd.r = diffuseColor.r;
	outMaterial.tex_Kd.g = diffuseColor.g;
	outMaterial.tex_Kd.b = diffuseColor.b;
	outMaterial.tex_Ks.r = specularColor.r;
	outMaterial.tex_Ks.g = specularColor.g;
	outMaterial.tex_Ks.b = specularColor.b;
	outMaterial.tex_d = ambientColor.a;
	outMaterial.tex_Ns = shininess;
	if (outMaterial.tex_Ks.r == 0 && outMaterial.tex_Ks.g == 0 && outMaterial.tex_Ks.b == 0)
	{
		outMaterial.tex_illum = 1;
	}
	else
	{
		outMaterial.tex_illum = 2;
	}
}

bool cc2smReader::getPclCloud2(ccGenericMesh* mesh, PCLCloud& cloud) const {
    unsigned int triNum = mesh->size();
    if (triNum <= 0) {
        CVLog::Warning("[cc2smReader::getPclCloud2] No triangles found!");
        return false;
    }

    std::size_t dimension = static_cast<std::size_t>(
            mesh->getTriangleVertIndexes(0)->getDimension());

    const ccGenericPointCloud::VisibilityTableType& verticesVisibility =
            mesh->getAssociatedCloud()->getTheVisibilityArray();
    bool visFiltering =
            (verticesVisibility.size() >= mesh->getAssociatedCloud()->size());

    bool showSF = mesh->hasDisplayedScalarField() && mesh->sfShown();
    bool showColors = showSF || (mesh->hasColors() && mesh->colorsShown());

    // per-triangle normals?
    bool showTriNormals = (mesh->hasTriNormals() && mesh->triNormsShown());
    // fix 'showNorms'
    bool showNorms = showTriNormals || (mesh->hasNormals() && mesh->normalsShown());

    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    {
        xyz_cloud->points.resize(static_cast<std::size_t>(triNum) * dimension);
        xyz_cloud->width = xyz_cloud->size();
        xyz_cloud->height = 1;
        xyz_cloud->is_dense = true;
    }

    pcl::PointCloud<pcl::RGB>::Ptr rgb_cloud = nullptr;
    if (showColors) {
        rgb_cloud.reset(new pcl::PointCloud<pcl::RGB>());
        rgb_cloud->points.resize(static_cast<std::size_t>(triNum) * dimension);
        rgb_cloud->width = rgb_cloud->size();
        rgb_cloud->height = 1;
        rgb_cloud->is_dense = true;
    }

    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud = nullptr;
    if (showNorms) {
        normal_cloud.reset(new pcl::PointCloud<pcl::Normal>());
        normal_cloud->resize(static_cast<std::size_t>(triNum) * dimension);
        normal_cloud->width = xyz_cloud->size();
        normal_cloud->height = 1;
        normal_cloud->is_dense = true;
    }

    // per-triangle normals
    const NormsIndexesTableType* triNormals = mesh->getTriNormsTable();
    // materials
    const ccMaterialSet* materials = mesh->getMaterialSet();

    // in the case we need normals (i.e. lighting)
    NormsIndexesTableType* normalsIndexesTable = nullptr;
    ccNormalVectors* compressedNormals = nullptr;
    if (showNorms) {
        //assert(m_cc_cloud->isA(CV_TYPES::POINT_CLOUD));
        normalsIndexesTable = m_cc_cloud->normals();
        compressedNormals = ccNormalVectors::GetUniqueInstance();
    }

    // current vertex normal
    const PointCoordinateType* N1 = nullptr;
    const PointCoordinateType* N2 = nullptr;
    const PointCoordinateType* N3 = nullptr;

    // vertices visibility
    for (unsigned n = 0; n < triNum; ++n) {
        const CVLib::VerticesIndexes* tsi = mesh->getTriangleVertIndexes(n);
        if (visFiltering) {
            // we skip the triangle if at least one vertex is hidden
            if ((verticesVisibility[tsi->i1] != POINT_VISIBLE) ||
                (verticesVisibility[tsi->i2] != POINT_VISIBLE) ||
                (verticesVisibility[tsi->i3] != POINT_VISIBLE))
                continue;
        }

        // First get the xyz information
        for (std::size_t vertexIndex = 0; vertexIndex < dimension;
             vertexIndex++) {
            (*xyz_cloud)[n * dimension + vertexIndex].x = static_cast<float>(
                    m_cc_cloud->getPoint(tsi->i[vertexIndex])->x);
            (*xyz_cloud)[n * dimension + vertexIndex].y = static_cast<float>(
                    m_cc_cloud->getPoint(tsi->i[vertexIndex])->y);
            (*xyz_cloud)[n * dimension + vertexIndex].z = static_cast<float>(
                    m_cc_cloud->getPoint(tsi->i[vertexIndex])->z);
        }

        // Then the color information, if any
        if (showSF) {
            for (std::size_t vertexIndex = 0; vertexIndex < dimension;
                 vertexIndex++) {
                // individual component copy due to different memory layout
                const ecvColor::Rgb* rgb =
                        m_cc_cloud->getCurrentDisplayedScalarField()
                                ->getValueColor(tsi->i[vertexIndex]);
                (*rgb_cloud)[n * dimension + vertexIndex].r = rgb->r;
                (*rgb_cloud)[n * dimension + vertexIndex].g = rgb->g;
                (*rgb_cloud)[n * dimension + vertexIndex].b = rgb->b;
                (*rgb_cloud)[n * dimension + vertexIndex].a = 255;
            }
        } else if (showColors) {
            for (std::size_t vertexIndex = 0; vertexIndex < dimension;
                 vertexIndex++) {
                // individual component copy due to different memory layout
                const ecvColor::Rgb* rgb =
                        &m_cc_cloud->rgbColors()->at(tsi->i[vertexIndex]);
                (*rgb_cloud)[n * dimension + vertexIndex].r = rgb->r;
                (*rgb_cloud)[n * dimension + vertexIndex].g = rgb->g;
                (*rgb_cloud)[n * dimension + vertexIndex].b = rgb->b;
                (*rgb_cloud)[n * dimension + vertexIndex].a = 255;
            }
        }

        // Then handle the normals, if any
        if (showNorms) {
            if (showTriNormals) {
                assert(triNormals);
                int n1 = 0;
                int n2 = 0;
                int n3 = 0;
                mesh->getTriangleNormalIndexes(n, n1, n2, n3);
                N1 = (n1 >= 0 ? ccNormalVectors::GetNormal(triNormals->at(n1)).u
                              : nullptr);
                N2 = (n1 == n2 ? N1
                               : n1 >= 0 ? ccNormalVectors::GetNormal(
                                                   triNormals->at(n2))
                                                   .u
                                         : nullptr);
                N3 = (n1 == n3 ? N1
                               : n3 >= 0 ? ccNormalVectors::GetNormal(
                                                   triNormals->at(n3))
                                                   .u
                                         : nullptr);
            } else {
                N1 = compressedNormals
                             ->getNormal(normalsIndexesTable->at(tsi->i1))
                             .u;
                N2 = compressedNormals
                             ->getNormal(normalsIndexesTable->at(tsi->i2))
                             .u;
                N3 = compressedNormals
                             ->getNormal(normalsIndexesTable->at(tsi->i3))
                             .u;
            }

            (*normal_cloud)[n * dimension + 0].normal_x = N1[0];
            (*normal_cloud)[n * dimension + 0].normal_y = N1[1];
            (*normal_cloud)[n * dimension + 0].normal_z = N1[2];
            (*normal_cloud)[n * dimension + 1].normal_x = N2[0];
            (*normal_cloud)[n * dimension + 1].normal_y = N2[1];
            (*normal_cloud)[n * dimension + 1].normal_z = N2[2];
            (*normal_cloud)[n * dimension + 2].normal_x = N3[0];
            (*normal_cloud)[n * dimension + 2].normal_y = N3[1];
            (*normal_cloud)[n * dimension + 2].normal_z = N3[2];
        }

    }

    // And put it in the mesh cloud
    {
        // points
        TO_PCL_CLOUD(*xyz_cloud, cloud);

        // colors
        if (showColors) {
            PCLCloud rgb_cloud2;
            TO_PCL_CLOUD(*rgb_cloud, rgb_cloud2);
            PCLCloud aux;
            pcl::concatenateFields(rgb_cloud2, cloud, aux);
            cloud = aux;
        }

        // normals
        if (showNorms) {
            PCLCloud normal_cloud2;
            TO_PCL_CLOUD(*normal_cloud, normal_cloud2);
            PCLCloud aux;
            pcl::concatenateFields(normal_cloud2, cloud, aux);
            cloud = aux;
        }
    }

	return true;
}

PCLTextureMesh::Ptr cc2smReader::getPclTextureMesh(ccGenericMesh* mesh) {
	if (!mesh) return nullptr;

	//materials & textures
	bool applyMaterials = (mesh->hasMaterials() && mesh->materialsShown());
	bool lodEnabled = false;
	bool showTextures = (mesh->hasTextures() && mesh->materialsShown() && !lodEnabled);

	if (applyMaterials || showTextures)
	{
		unsigned int triNum = mesh->size();
        if (triNum <= 0) {
            CVLog::Warning("[cc2smReader::getPclTextureMesh] No triangles found!");
            return nullptr;
        }

		PCLTextureMesh::Ptr textureMesh(new PCLTextureMesh);
        if (!getPclCloud2(mesh, textureMesh->cloud)) {
            CVLog::Warning("[cc2smReader::getPclTextureMesh] Failed to get "
                    "pcl::PCLPointCloud2!");
            return nullptr;
        }

		const ccGenericPointCloud::VisibilityTableType& verticesVisibility = 
			mesh->getAssociatedCloud()->getTheVisibilityArray();
		bool visFiltering = (verticesVisibility.size() >= mesh->getAssociatedCloud()->size());

		// materials
        const ccMaterialSet* materials = mesh->getMaterialSet();

        // loop on all triangles
		int lasMtlIndex = -1;
		unsigned int currentTexID = 0;

		//vertices visibility
        for (unsigned n = 0; n < triNum; ++n)
		{
			const CVLib::VerticesIndexes* tsi = mesh->getTriangleVertIndexes(n);
			if (visFiltering)
			{
				//we skip the triangle if at least one vertex is hidden
				if ((verticesVisibility[tsi->i1] != POINT_VISIBLE) ||
					(verticesVisibility[tsi->i2] != POINT_VISIBLE) ||
					(verticesVisibility[tsi->i3] != POINT_VISIBLE))
					continue;
			}

			assert(materials);
			int newMatlIndex = mesh->getTriangleMtlIndex(n);
			// do we need to change material?
			if (lasMtlIndex != newMatlIndex)
			{
				assert(newMatlIndex < static_cast<int>(materials->size()));
				if (showTextures)
				{
					if (currentTexID)
					{
						currentTexID = 0;
					}

					if (newMatlIndex >= 0)
					{
						currentTexID = materials->at(newMatlIndex)->getTextureID();
					}
				}

				if (newMatlIndex >= 0) {
					textureMesh->tex_polygons.push_back(std::vector<pcl::Vertices>());
					textureMesh->tex_materials.push_back(PCLMaterial());
					textureMesh->tex_coordinates.push_back(std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> >());
					getMaterial((*materials)[newMatlIndex], textureMesh->tex_materials.back());
				} else { // if we don't have any current material, we apply default one
					textureMesh->tex_polygons.push_back(std::vector<pcl::Vertices>());
					textureMesh->tex_materials.push_back(PCLMaterial());
					textureMesh->tex_coordinates.push_back(std::vector< Eigen::Vector2f, Eigen::aligned_allocator< Eigen::Vector2f>>());
					ccMaterial::Shared defaultMaterial(new ccMaterial("default"));
					getMaterial(defaultMaterial, textureMesh->tex_materials.back());
				}

				lasMtlIndex = newMatlIndex;
			}

			// get the texture coordinates information
			if (showTextures && !textureMesh->tex_coordinates.empty())
			{
				//current vertex texture coordinates
				TexCoords2D* Tx1 = nullptr;
				TexCoords2D* Tx2 = nullptr;
				TexCoords2D* Tx3 = nullptr;
				mesh->getTriangleTexCoordinates(n, Tx1, Tx2, Tx3);
				textureMesh->tex_coordinates.back().push_back(Eigen::Vector2f(Tx1->tx, Tx1->ty));
				textureMesh->tex_coordinates.back().push_back(Eigen::Vector2f(Tx2->tx, Tx2->ty));
				textureMesh->tex_coordinates.back().push_back(Eigen::Vector2f(Tx3->tx, Tx3->ty));
			}

			pcl::Vertices tri;
			tri.vertices.push_back(n * 3 + 0);
            tri.vertices.push_back(n * 3 + 1);
            tri.vertices.push_back(n * 3 + 2);

			if (!textureMesh->tex_polygons.empty())
			{
				textureMesh->tex_polygons.back().push_back(tri);
			}
		}

		// check 
		{
			// no texture materials --> exit
			if (textureMesh->tex_materials.size() == 0)
			{
				CVLog::Warning("[cc2smReader::getPclTextureMesh] No textures found!");
				return nullptr;
			}
			// polygons are mapped to texture materials
			if (textureMesh->tex_materials.size() != textureMesh->tex_polygons.size())
			{
				CVLog::Warning("[cc2smReader::getPclTextureMesh] Materials number %lu differs from polygons number %lu!",
					textureMesh->tex_materials.size(), textureMesh->tex_polygons.size());
				return nullptr;
			}
			// each texture material should have its coordinates set
			if (textureMesh->tex_materials.size() != textureMesh->tex_coordinates.size())
			{
				CVLog::Warning("[cc2smReader::getPclTextureMesh] Coordinates number %lu differs from materials number %lu!",
					textureMesh->tex_coordinates.size(), textureMesh->tex_materials.size());
				return nullptr;
			}
		}
		return textureMesh;
	}
	else
	{
		CVLog::Warning("[cc2smReader::getPclTextureMesh] this mesh has no material and texture and please try polygonMesh other than textureMesh");
		return nullptr;
	}
}

PCLPolygon::Ptr cc2smReader::getPclPolygon(ccPolyline * polyline) const
{
	PCLPolygon::Ptr pclPolygon(new PCLPolygon);
	pcl::PointCloud<PointT>::Ptr cloud_xyz(new pcl::PointCloud<PointT>);
	if (polyline->size() < 2)
	{
		return nullptr;
	}

	cloud_xyz->resize(polyline->size());
	
	for (unsigned i = 0; i < polyline->size(); ++i)
	{
		const CCVector3 *pp = polyline->getPoint(i);
		CCVector3d output3D;
		if (polyline->is2DMode()) {
			PCLDisplayTools::TheInstance()->toWorldPoint(*pp, output3D);
		}
		else {
			output3D = CCVector3d::fromArray(pp->u);
		}
		
		cloud_xyz->points[i].x = output3D.x;
		cloud_xyz->points[i].y = output3D.y;
		cloud_xyz->points[i].z = output3D.z;
	}

	pclPolygon->setContour(*cloud_xyz);

	return pclPolygon;
}