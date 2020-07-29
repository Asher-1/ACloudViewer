// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "PointCloudIO.h"
#include <iostream>
#include <unordered_map>

#include <Console.h>
#include <CVTools.h>
#include <FileSystem.h>
#include <ecvPointCloud.h>
#include <FileIOFilter.h>

#include <QFileInfo>

namespace cloudViewer {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, ccPointCloud &, bool)>>
        file_extension_to_pointcloud_read_function{
				{"xyz", ReadPointCloudFromXYZ},
				{"txt", ReadPointCloudFromXYZ},
				{"xyzn", ReadPointCloudFromXYZN},
				{"xyzrgb", ReadPointCloudFromXYZRGB},
				{"ply", ReadPointCloudFromPLY},
				{"pcd", ReadPointCloudFromPCD},
				//{"pts", ReadPointCloudFromPTS},
                {"pts", AutoReadPointCloud},
                {"e57", AutoReadPointCloud},
                {"fbx", AutoReadPointCloud},
                {"ptx", AutoReadPointCloud},
				{"vtk", AutoReadPointCloud},
                {"bin", AutoReadPointCloud},
				{"csv", AutoReadPointCloud},
        };

static const std::unordered_map<std::string,
                                std::function<bool(const std::string &,
                                                   const ccPointCloud &,
                                                   const bool,
                                                   const bool,
                                                   const bool)>>
        file_extension_to_pointcloud_write_function{
				{"xyz", WritePointCloudToXYZ},
				{"txt", WritePointCloudToXYZ},
				{"xyzn", WritePointCloudToXYZN},
				{"xyzrgb", WritePointCloudToXYZRGB},
				{"ply", WritePointCloudToPLY},
				{"pcd", WritePointCloudToPCD},
				//{"pts", WritePointCloudToPTS},
                {"pts", AutoWritePointCloud},
                {"e57", AutoWritePointCloud},
                {"fbx", AutoWritePointCloud},
                {"ptx", AutoWritePointCloud},
                {"vtk", AutoWritePointCloud},
                {"bin", AutoWritePointCloud},
				{"csv", AutoWritePointCloud},
        };

}  // unnamed namespace

namespace io {

std::shared_ptr<ccPointCloud> CreatePointCloudFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<ccPointCloud>("pointCloud");
    ReadPointCloud(filename, *pointcloud, format, print_progress);
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    ccPointCloud &pointcloud,
                    const std::string &format,
                    bool remove_nan_points,
                    bool remove_infinite_points,
                    bool print_progress) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                CVLib::utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }

    std::cout << "Format = " << format << std::endl;
    std::cout << "Extension = " << filename_ext << std::endl;

    if (filename_ext.empty()) {
        CVLib::utility::LogWarning(
                "Read ccPointCloud failed: unknown file extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_pointcloud_read_function.find(filename_ext);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        CVLib::utility::LogWarning(
                "Read ccPointCloud failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, print_progress);
    CVLib::utility::LogDebug("Read ccPointCloud: {:d} vertices.",
                      (int)pointcloud.size());
    if (remove_nan_points || remove_infinite_points) {
        pointcloud.removeNonFinitePoints(remove_nan_points,
                                         remove_infinite_points);
    }
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const ccPointCloud &pointcloud,
                     bool write_ascii /* = false*/,
                     bool compressed /* = false*/,
                     bool print_progress) {
    std::string filename_ext =
            CVLib::utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        CVLib::utility::LogWarning(
                "Write ccPointCloud failed: unknown file extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_pointcloud_write_function.find(filename_ext);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        CVLib::utility::LogWarning(
                "Write ccPointCloud failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, write_ascii,
                                   compressed, print_progress);
    CVLib::utility::LogDebug("Write ccPointCloud: {:d} vertices.",
                      (int)pointcloud.size());
    return success;
}


bool AutoReadPointCloud(const std::string & filename,
	ccPointCloud & pointcloud, bool print_progress)
{
	//to use the same 'global shift' for multiple files
	CCVector3d loadCoordinatesShift(0, 0, 0);
	bool loadCoordinatesTransEnabled = false;

	FileIOFilter::LoadParameters parameters;
	{
		parameters.alwaysDisplayLoadDialog = print_progress;
		parameters.shiftHandlingMode = ecvGlobalShiftManager::NO_DIALOG;
		parameters.coordinatesShift = &loadCoordinatesShift;
		parameters.coordinatesShiftEnabled = &loadCoordinatesTransEnabled;
		parameters.parentWidget = nullptr;
		parameters.autoComputeNormals = false;
	}

	//the same for 'addToDB' (if the first one is not supported, or if the scale remains too big)
	CCVector3d addCoordinatesShift(0, 0, 0);

	if (FileIOFilter::GetFilters().empty())
	{
		FileIOFilter::InitInternalFilters();
	}

	FileIOFilter::ResetSesionCounter();

	CC_FILE_ERROR result = CC_FERR_NO_ERROR;
	FileIOFilter::Shared filter(nullptr);

	//load file
	auto container = std::make_shared<ccHObject>();

	while (true)
	{
		//look for file extension (we trust Qt on this task)
		QString file = CVTools::toQString(filename);
		QString extension = QFileInfo(file).suffix();
		if (extension.isEmpty())
		{
			CVLib::utility::LogWarning("[Load] Can't guess file format: no file extension");
			result = CC_FERR_CONSOLE_ERROR;
			break;
		}
		else
		{
			//convert extension to file format
			filter = FileIOFilter::FindBestFilterForExtension(extension);

			//unknown extension?
			if (!filter)
			{
				CVLib::utility::LogWarning(
					"[Load] Can't guess file format: unhandled file extension '%s'", 
					extension.toStdString().c_str());
				result = CC_FERR_CONSOLE_ERROR;
				break;
			}

			//check file existence
			QFileInfo fi(file);
			if (!fi.exists())
			{
				CVLib::utility::LogWarning(
					"[Load] File '%s' doesn't exist!", file.toStdString().c_str());
				result = CC_FERR_CONSOLE_ERROR;
				break;
			}

			//we start a new 'action' inside the current sessions
			unsigned sessionCounter = FileIOFilter::IncreaseSesionCounter();
			parameters.sessionStart = (sessionCounter == 1);

			try
			{
				result = filter->loadFile(file, *container, parameters);
			}
			catch (const std::exception& e)
			{
				CVLib::utility::LogWarning(
					"[I/O] CC has caught an exception while loading file '%s'!", 
					file.toStdString().c_str());
				CVLib::utility::LogWarning("[I/O] Exception: %s", e.what());
				if (container)
				{
					container->removeAllChildren();
				}
				result = CC_FERR_CONSOLE_ERROR;
			}
			catch (...)
			{
				CVLib::utility::LogWarning(
					"[I/O] CC has caught an unhandled exception while loading file '%s'",
					file.toStdString().c_str());
				if (container)
				{
					container->removeAllChildren();
				}
				result = CC_FERR_CONSOLE_ERROR;
			}

			if (result != CC_FERR_NO_ERROR)
			{
				FileIOFilter::DisplayErrorMessage(result, "loading", fi.baseName());
			}

			unsigned childCount = container->getChildrenNumber();
			if (childCount != 0)
			{
				//we set the main container name as the full filename (with path)
				container->setName(QString("%1 (%2)").arg(fi.fileName(), fi.absolutePath()));
				for (unsigned i = 0; i < childCount; ++i)
				{
					ccHObject* child = container->getChild(i);
					child->setBaseName(fi.baseName());
					child->setFullPath(file);
					QString newName = child->getName();
					if (newName.startsWith("unnamed"))
					{
						//we automatically replace occurrences of 'unnamed' in entities names by the base filename (no path, no extension)
						newName.replace(QString("unnamed"), fi.baseName());
						child->setName(newName);
					}
				}
			}
			else
			{
				result = CC_FERR_NO_LOAD;
				break;
			}

			if (container)
			{
				//disable the normals on all loaded clouds!
				ccHObject::Container clouds;
				container->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD);
				if (clouds.size() != 1)
				{
					container->removeAllChildren();
					result = CC_FERR_BAD_ENTITY_TYPE;
					break;
				}
			}
		}

		break;
	}

	if (result == CC_FERR_NO_ERROR)
	{
		ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(container->getChild(0));
		if (!cloud) return false;

		pointcloud.append(cloud, pointcloud.size());
		container->removeAllChildren();
		return true;
	}
	else
		return false;

}

bool AutoWritePointCloud(const std::string & filename,
	const ccPointCloud & pointcloud, bool write_ascii,
	bool compressed, bool print_progress)
{
	FileIOFilter::SaveParameters parameters;
	{
		parameters.alwaysDisplaySaveDialog = print_progress;
		parameters.parentWidget = nullptr;
	}

	if (FileIOFilter::GetFilters().empty())
	{
		FileIOFilter::InitInternalFilters();
	}

	FileIOFilter::ResetSesionCounter();

	CC_FILE_ERROR result = CC_FERR_NO_ERROR;
	FileIOFilter::Shared filter(nullptr);

	// save file
	while (true)
	{
		//look for file extension (we trust Qt on this task)
		QString completeFileName = CVTools::toQString(filename);

		//if the file name has no extension, we had a default one!
		
		if (QFileInfo(completeFileName).suffix().isEmpty())
			completeFileName += QString(".%1").arg(filter->getDefaultExtension());
		
		QString extension = QFileInfo(completeFileName).suffix();
		CC_FILE_ERROR result = CC_FERR_NO_ERROR;
		{
			//convert extension to file format
			filter = FileIOFilter::FindBestFilterForExtension(extension);

			//unknown extension?
			if (!filter)
			{
				CVLib::utility::LogWarning(
					"[Load] Can't guess file format: unhandled file extension '%s'", 
					CVTools::fromQString(extension).c_str());
				result = CC_FERR_CONSOLE_ERROR;
				break;
			}

			try
			{
				result = filter->saveToFile(const_cast<ccPointCloud*>(&pointcloud), completeFileName, parameters);
			}
			catch (...)
			{
				CVLib::utility::LogWarning(
					"[I/O] CC has caught an unhandled exception while saving file '%s'", 
					CVTools::fromQString(completeFileName).c_str());
				result = CC_FERR_CONSOLE_ERROR;
			}

			if (result != CC_FERR_NO_ERROR)
			{
				FileIOFilter::DisplayErrorMessage(result, "saving", completeFileName);
			}

		}

		break;
	}

	return result == CC_FERR_NO_ERROR;

}

}  // namespace io
}  // namespace cloudViewer
