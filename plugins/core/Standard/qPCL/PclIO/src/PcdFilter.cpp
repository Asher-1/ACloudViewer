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
//#                    COPYRIGHT: CLOUDVIEWER  project                     #
//#                                                                        #
//##########################################################################

#include "PcdFilter.h"
#include "ui_savePCDFileDlg.h"

// PclUtils
#include <PCLConv.h>
#include <sm2cc.h>
#include <cc2sm.h>

// CV_CORE_LIB
#include <CVTools.h>

// ECV_DB_LIB
#include <ecvGBLSensor.h>
#include <ecvPointCloud.h>
#include <ecvHObjectCaster.h>

//Qt
#include <QSettings>
#include <QFileInfo>

//Boost
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>

//System
#include <iostream>

//pcl
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>


PcdFilter::PcdFilter()
	: FileIOFilter({
					"_Point Cloud Library Filter",
					13.0f,	// priority
					QStringList{ "pcd" },
					"pcd",
					QStringList{ "Point Cloud Library cloud (*.pcd)" },
					QStringList{ "Point Cloud Library cloud (*.pcd)" },
					Import | Export
		})
{
}

bool PcdFilter::canSave(CV_CLASS_ENUM type, bool& multiple, bool& exclusive) const
{
	//only one cloud per file
	if (type == CV_TYPES::POINT_CLOUD)
	{
		multiple = false;
		exclusive = true;
		return true;
	}

	return false;
}

//! PCD File Save dialog
class SavePCDFileDialog : public QDialog, public Ui::SavePCDFileDlg
{
public:
	//! Default constructor
	explicit SavePCDFileDialog(QWidget* parent = nullptr)
		: QDialog(parent)
		, Ui::SavePCDFileDlg()
	{
		setupUi(this);
	}
};

CC_FILE_ERROR PcdFilter::saveToFile(ccHObject* entity,
                                    const QString& filename,
                                    const SaveParameters& parameters)
{
	if (!entity || filename.isEmpty())
		return CC_FERR_BAD_ARGUMENT;

	//the cloud to save
	ccPointCloud* ccCloud = ccHObjectCaster::ToPointCloud(entity);
	if (!ccCloud)
	{
		CVLog::Warning("[PCL] This filter can only save one cloud at a time!");
		return CC_FERR_BAD_ENTITY_TYPE;
	}

	// display pcd save dialog
	QSettings settings("save pcd");
	settings.beginGroup("SavePcd");

	SavePCDFileDialog spfDlg(nullptr);
	bool saveOriginOrientation = settings.value("saveOriginOrientation", spfDlg.saveOriginOrientationCheckBox->isChecked()).toBool();
	bool saveBinary = settings.value("SavePCDBinary", spfDlg.saveBinaryCheckBox->isChecked()).toBool();
	bool compressedMode = settings.value("Compressed", spfDlg.saveCompressedCheckBox->isChecked()).toBool();
	compressedMode = saveBinary && compressedMode ? true : false;
	{
		spfDlg.saveOriginOrientationCheckBox->setChecked(saveOriginOrientation);
		spfDlg.saveBinaryCheckBox->setChecked(saveBinary);
		spfDlg.saveCompressedCheckBox->setChecked(compressedMode);

		if (!spfDlg.exec())
			return CC_FERR_CANCELED_BY_USER;

		saveOriginOrientation = spfDlg.saveOriginOrientationCheckBox->isChecked();
		saveBinary = spfDlg.saveBinaryCheckBox->isChecked();
		compressedMode = spfDlg.saveCompressedCheckBox->isChecked();
		settings.setValue("saveOriginOrientation", saveOriginOrientation);
		settings.setValue("SavePCDBinary", saveBinary);
		settings.setValue("Compressed", saveBinary && compressedMode ? true : false);

		settings.endGroup();
	}
	
	PCLCloud::Ptr pclCloud = cc2smReader(ccCloud).getAsSM();
	if (!pclCloud)
	{
		return CC_FERR_THIRD_PARTY_LIB_FAILURE;
	}

	if (saveOriginOrientation)
	{
		//search for a sensor as child (we take the first if there are several of them)
        ccSensor* sensor(nullptr);
		{
			for (unsigned i = 0; i < ccCloud->getChildrenNumber(); ++i)
			{
				ccHObject* child = ccCloud->getChild(i);

				//try to cast to a ccSensor
				sensor = ccHObjectCaster::ToSensor(child);
				if (sensor)
					break;
			}
		}

		Eigen::Vector4f pos;
		Eigen::Quaternionf ori;
		if (!sensor)
		{
			//we append to the cloud null sensor informations
			pos = Eigen::Vector4f::Zero();
			ori = Eigen::Quaternionf::Identity();
		}
		else
		{
			//we get out valid sensor informations
			ccGLMatrix mat = sensor->getRigidTransformation();
			CCVector3 trans = mat.getTranslationAsVec3D();
			pos(0) = trans.x;
			pos(1) = trans.y;
			pos(2) = trans.z;

			//also the rotation
			Eigen::Matrix3f eigrot;
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					eigrot(i, j) = mat.getColumn(j)[i];

			//now translate to a quaternion notation
			ori = Eigen::Quaternionf(eigrot);
		}

        if (ccCloud->size() == 0)
        {
            pcl::PCDWriter w;
            QFile file(filename);
            if (!file.open(QFile::WriteOnly | QFile::Truncate))
                return CC_FERR_WRITING;
            QTextStream stream(&file);

            if (compressedMode) {
                stream << QString(w.generateHeaderBinaryCompressed(*pclCloud, pos, ori).c_str()) << "DATA binary\n";
            } else {
                stream << QString(w.generateHeaderBinary(*pclCloud, pos, ori).c_str()) << "DATA binary\n";
            }
            return CC_FERR_NO_ERROR;
        }

		if (compressedMode)
		{
			pcl::PCDWriter w;
			if (w.writeBinaryCompressed( /*qPrintable*/CVTools::FromQString(filename), *pclCloud, pos, ori) < 0)
			{
				return CC_FERR_THIRD_PARTY_LIB_FAILURE;
			}
		}
		else
		{
			if (pcl::io::savePCDFile( /*qPrintable*/CVTools::FromQString(filename), *pclCloud, pos, ori, saveBinary) < 0) //DGM: warning, toStdString doesn't preserve "local" characters
			{
				return CC_FERR_THIRD_PARTY_LIB_FAILURE;
			}
		}
	}
	else
	{
        if (ccCloud->size() == 0)
        {
            pcl::PCDWriter w;
            QFile file(filename);
            if (!file.open(QFile::WriteOnly | QFile::Truncate))
                return CC_FERR_WRITING;
            QTextStream stream(&file);

            Eigen::Vector4f pos;
            Eigen::Quaternionf ori;

            if (compressedMode) {
                stream << QString(w.generateHeaderBinaryCompressed(*pclCloud, pos, ori).c_str()) << "DATA binary\n";
            } else {
                stream << QString(w.generateHeaderBinary(*pclCloud, pos, ori).c_str()) << "DATA binary\n";
            }
            return CC_FERR_NO_ERROR;
        }

		bool hasColor = ccCloud->hasColors();
		bool hasNormals = ccCloud->hasNormals();
		if (hasColor && !hasNormals)
		{
			PointCloudRGB::Ptr rgbCloud(new PointCloudRGB);
			FROM_PCL_CLOUD(*pclCloud, *rgbCloud);
			if (!rgbCloud)
			{
				return CC_FERR_THIRD_PARTY_LIB_FAILURE;
			}
			if (compressedMode)
			{
				if (pcl::io::savePCDFileBinaryCompressed(CVTools::FromQString(filename), *rgbCloud) < 0) //DGM: warning, toStdString doesn't preserve "local" characters
				{
					return CC_FERR_THIRD_PARTY_LIB_FAILURE;
				}
			}
			else
			{
				if (pcl::io::savePCDFile(CVTools::FromQString(filename), *rgbCloud, saveBinary) < 0) //DGM: warning, toStdString doesn't preserve "local" characters
				{
					return CC_FERR_THIRD_PARTY_LIB_FAILURE;
				}
			}
		}
		else if (!hasColor && hasNormals)
		{
			PointCloudNormal::Ptr normalCloud(new PointCloudNormal);
			FROM_PCL_CLOUD(*pclCloud, *normalCloud);
			if (!normalCloud)
			{
				return CC_FERR_THIRD_PARTY_LIB_FAILURE;
			}
			if (compressedMode)
			{
				if (pcl::io::savePCDFileBinaryCompressed(CVTools::FromQString(filename), *normalCloud) < 0) //DGM: warning, toStdString doesn't preserve "local" characters
				{
					return CC_FERR_THIRD_PARTY_LIB_FAILURE;
				}
			}
			else
			{
				if (pcl::io::savePCDFile(CVTools::FromQString(filename), *normalCloud, saveBinary) < 0) //DGM: warning, toStdString doesn't preserve "local" characters
				{
					return CC_FERR_THIRD_PARTY_LIB_FAILURE;
				}
			}
		}
		else if (hasColor && hasNormals)
		{
			PointCloudRGBNormal::Ptr rgbNormalCloud(new PointCloudRGBNormal);
			FROM_PCL_CLOUD(*pclCloud, *rgbNormalCloud);
			if (!rgbNormalCloud)
			{
				return CC_FERR_THIRD_PARTY_LIB_FAILURE;
			}
			if (compressedMode)
			{
				if (pcl::io::savePCDFileBinaryCompressed(CVTools::FromQString(filename), *rgbNormalCloud) < 0) //DGM: warning, toStdString doesn't preserve "local" characters
				{
					return CC_FERR_THIRD_PARTY_LIB_FAILURE;
				}
			}
			else
			{
				if (pcl::io::savePCDFile(CVTools::FromQString(filename), *rgbNormalCloud, saveBinary) < 0) //DGM: warning, toStdString doesn't preserve "local" characters
				{
					return CC_FERR_THIRD_PARTY_LIB_FAILURE;
				}
			}
		}
		else  // just save xyz coordinates
		{
			PointCloudT::Ptr xyzCloud(new PointCloudT);
			FROM_PCL_CLOUD(*pclCloud, *xyzCloud);
			if (!xyzCloud)
			{
				return CC_FERR_THIRD_PARTY_LIB_FAILURE;
			}
			if (compressedMode)
			{
				if (pcl::io::savePCDFileBinaryCompressed(CVTools::FromQString(filename), *xyzCloud) < 0) //DGM: warning, toStdString doesn't preserve "local" characters
				{
					return CC_FERR_THIRD_PARTY_LIB_FAILURE;
				}
			}
			else
			{
				if (pcl::io::savePCDFile(CVTools::FromQString(filename), *xyzCloud, saveBinary) < 0) //DGM: warning, toStdString doesn't preserve "local" characters
				{
					return CC_FERR_THIRD_PARTY_LIB_FAILURE;
				}
			}
		}
	}

	return CC_FERR_NO_ERROR;
}

CC_FILE_ERROR PcdFilter::loadFile(const QString& filename, ccHObject& container, LoadParameters& parameters)
{
	Eigen::Vector4f origin;
	Eigen::Quaternionf orientation;
	int pcd_version;
	int data_type;
	unsigned int data_idx;
	size_t pointCount = -1;
    PCLCloud inputCloud;
	//Load the given file
	pcl::PCDReader p;

	const std::string& fileName = CVTools::FromQString(filename);

    if (p.readHeader(fileName, inputCloud, origin, orientation, pcd_version, data_type, data_idx) < 0)
    {
        return CC_FERR_THIRD_PARTY_LIB_FAILURE;
    }

    pointCount = inputCloud.width * inputCloud.height;
    CVLog::Print(QString("%1: Point Count: %2").arg(qPrintable(filename)).arg(pointCount));

    if (pointCount == 0)
    {
        return CC_FERR_NO_LOAD;
    }

    // DGM: warning, toStdString doesn't preserve "local" characters
    if (pcl::io::loadPCDFile(fileName, inputCloud, origin, orientation) < 0)
    {
        return CC_FERR_THIRD_PARTY_LIB_FAILURE;
    }

    ccPointCloud* ccCloud = nullptr;
    if (!inputCloud.is_dense) //data may contain NaNs --> remove them
    {
        //now we need to remove NaNs
        pcl::PassThrough<PCLCloud> passFilter;
        passFilter.setInputCloud(PCLCloud::Ptr(new PCLCloud(inputCloud)));

        PCLCloud filteredCloud;
        passFilter.filter(filteredCloud);

        ccCloud = pcl2cc::Convert(filteredCloud);
    }
    else
    {
        ccCloud = pcl2cc::Convert(inputCloud);
    }

    //convert to CC cloud
	if (!ccCloud)
	{
        CVLog::Warning("[PCL] An error occurred while converting PCD cloud to CloudViewer  cloud!");
		return CC_FERR_CONSOLE_ERROR;
	}
	ccCloud->setName(QStringLiteral("unnamed"));

    //now we construct a ccGBLSensor
    {
        // get orientation as rot matrix and copy it into a ccGLMatrix
        ccGLMatrix ccRot;
        {
            Eigen::Matrix3f eigrot = orientation.toRotationMatrix();
            float* X = ccRot.getColumn(0);
            float* Y = ccRot.getColumn(1);
            float* Z = ccRot.getColumn(2);

            X[0] = eigrot(0,0); X[1] = eigrot(1,0); X[2] = eigrot(2,0);
            Y[0] = eigrot(0,1); Y[1] = eigrot(1,1); Y[2] = eigrot(2,1);
            Z[0] = eigrot(0,2); Z[1] = eigrot(1,2); Z[2] = eigrot(2,2);

            ccRot.getColumn(3)[3] = 1.0f;
            ccRot.setTranslation(origin.data());
        }

        ccGBLSensor* sensor = new ccGBLSensor;
        sensor->setRigidTransformation(ccRot);
        sensor->setYawStep(static_cast<PointCoordinateType>(0.05));
        sensor->setPitchStep(static_cast<PointCoordinateType>(0.05));
        sensor->setVisible(true);
        //uncertainty to some default
        sensor->setUncertainty(static_cast<PointCoordinateType>(0.01));
        //graphic scale
        sensor->setGraphicScale(ccCloud->getOwnBB().getDiagNorm() / 10);

        //Compute parameters
        ccGenericPointCloud* pc = ccHObjectCaster::ToGenericPointCloud(ccCloud);
        sensor->computeAutoParameters(pc);

        sensor->setEnabled(false);

        ccCloud->addChild(sensor);
    }

    container.addChild(ccCloud);

    return CC_FERR_NO_ERROR;
}
