// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

// LOCAL
#include "AutoIO.h"

// CV_CORE_LIB
#include <CVTools.h>
#include <FileSystem.h>
#include <Logging.h>
#include <ProgressReporters.h>

// ECV_DB_LIB
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// ECV_IO_LIB
#include <FileIOFilter.h>

// SYSTEM
#include <unordered_map>

// QT
#include <QFileInfo>

namespace cloudViewer {
namespace io {

static const std::unordered_map<
        std::string,
        std::function<bool(
                const std::string&, ccHObject&, const ReadPointCloudOption&)>>
        file_extension_to_entity_read_function{
                {"bin", AutoReadEntity},  {"ply", AutoReadEntity},
                {"vtk", AutoReadEntity},  {"stl", AutoReadEntity},
                {"pcd", AutoReadEntity},  {"off", AutoReadEntity},
                {"dxf", AutoReadEntity},  {"txt", AutoReadEntity},
                {"las", AutoReadEntity},  {"laz", AutoReadEntity},
                {"mat", AutoReadEntity},  {"obj", AutoReadEntity},
                {"ptx", AutoReadEntity},  {"pt", AutoReadEntity},
                {"poly", AutoReadEntity}, {"shp", AutoReadEntity},
                {"sbf", AutoReadEntity},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string&,
                           const ccHObject&,
                           const WritePointCloudOption&)>>
        file_extension_to_entity_write_function{
                {"bin", AutoWriteEntity},  {"ply", AutoWriteEntity},
                {"vtk", AutoWriteEntity},  {"stl", AutoWriteEntity},
                {"pcd", AutoWriteEntity},  {"off", AutoWriteEntity},
                {"dxf", AutoWriteEntity},  {"txt", AutoWriteEntity},
                {"las", AutoWriteEntity},  {"laz", AutoWriteEntity},
                {"mat", AutoWriteEntity},  {"obj", AutoWriteEntity},
                {"ptx", AutoWriteEntity},  {"pt", AutoWriteEntity},
                {"poly", AutoWriteEntity}, {"shp", AutoWriteEntity},
                {"sbf", AutoWriteEntity},
        };

bool AutoReadEntity(const std::string& filename,
                    ccHObject& entity,
                    const ReadPointCloudOption& params) {
    // to use the same 'global shift' for multiple files
    CCVector3d loadCoordinatesShift(0, 0, 0);
    bool loadCoordinatesTransEnabled = false;

    FileIOFilter::LoadParameters parameters;
    {
        parameters.alwaysDisplayLoadDialog = params.print_progress;
        parameters.shiftHandlingMode = ecvGlobalShiftManager::NO_DIALOG;
        parameters.coordinatesShift = &loadCoordinatesShift;
        parameters.coordinatesShiftEnabled = &loadCoordinatesTransEnabled;
        parameters.parentWidget = nullptr;
        parameters.autoComputeNormals = false;
    }

    // the same for 'addToDB' (if the first one is not supported, or if the
    // scale remains too big)
    CCVector3d addCoordinatesShift(0, 0, 0);

    if (FileIOFilter::GetFilters().empty()) {
        FileIOFilter::InitInternalFilters();
    }

    FileIOFilter::ResetSesionCounter();

    CC_FILE_ERROR result = CC_FERR_NO_ERROR;
    FileIOFilter::Shared filter(nullptr);

    // load file
    auto container = cloudViewer::make_shared<ccHObject>();

    while (true) {
        // look for file extension (we trust Qt on this task)
        QString file = CVTools::ToQString(filename);
        QString extension = QFileInfo(file).suffix();
        if (extension.isEmpty()) {
            cloudViewer::utility::LogWarning(
                    "[Load] Can't guess file format: no file extension");
            result = CC_FERR_CONSOLE_ERROR;
            break;
        } else {
            // convert extension to file format
            filter = FileIOFilter::FindBestFilterForExtension(extension);

            // unknown extension?
            if (!filter) {
                cloudViewer::utility::LogWarning(
                        "[Load] Can't guess file format: unhandled file "
                        "extension '%s'",
                        extension.toStdString().c_str());
                result = CC_FERR_CONSOLE_ERROR;
                break;
            }

            // check file existence
            QFileInfo fi(file);
            if (!fi.exists()) {
                cloudViewer::utility::LogWarning(
                        "[Load] File '%s' doesn't exist!",
                        file.toStdString().c_str());
                result = CC_FERR_CONSOLE_ERROR;
                break;
            }

            // we start a new 'action' inside the current sessions
            unsigned sessionCounter = FileIOFilter::IncreaseSesionCounter();
            parameters.sessionStart = (sessionCounter == 1);

            try {
                if (entity.isA(CV_TYPES::HIERARCHY_OBJECT)) {
                    result = filter->loadFile(file, entity, parameters);
                } else {
                    result = filter->loadFile(file, *container, parameters);
                }
            } catch (const std::exception& e) {
                cloudViewer::utility::LogWarning(
                        "[I/O] CC has caught an exception while loading file "
                        "'%s'!",
                        file.toStdString().c_str());
                cloudViewer::utility::LogWarning("[I/O] Exception: %s",
                                                 e.what());
                if (container) {
                    container->removeAllChildren();
                }
                result = CC_FERR_CONSOLE_ERROR;
            } catch (...) {
                cloudViewer::utility::LogWarning(
                        "[I/O] CC has caught an unhandled exception while "
                        "loading file '%s'",
                        file.toStdString().c_str());
                if (container) {
                    container->removeAllChildren();
                }
                result = CC_FERR_CONSOLE_ERROR;
            }

            if (result != CC_FERR_NO_ERROR) {
                FileIOFilter::DisplayErrorMessage(result, "loading",
                                                  fi.baseName());
            }

            unsigned childCount = container->getChildrenNumber();
            if (childCount != 0) {
                // we set the main container name as the full filename (with
                // path)
                container->setName(QString("%1 (%2)").arg(fi.fileName(),
                                                          fi.absolutePath()));
                for (unsigned i = 0; i < childCount; ++i) {
                    ccHObject* child = container->getChild(i);
                    child->setBaseName(fi.baseName());
                    child->setFullPath(file);
                    QString newName = child->getName();
                    if (newName.startsWith("unnamed")) {
                        // we automatically replace occurrences of 'unnamed' in
                        // entities names by the base filename (no path, no
                        // extension)
                        newName.replace(QString("unnamed"), fi.baseName());
                        child->setName(newName);
                    }
                }
            } else {
                result = CC_FERR_NO_LOAD;
                break;
            }
        }

        break;
    }

    bool successFlag = true;
    if (result == CC_FERR_NO_ERROR) {
        if (entity.isKindOf(CV_TYPES::POINT_CLOUD)) {
            ccPointCloud* outCloud = ccHObjectCaster::ToPointCloud(&entity);
            for (unsigned i = 0; i < container->getChildrenNumber(); ++i) {
                ccPointCloud* cloud =
                        ccHObjectCaster::ToPointCloud(container->getChild(i));
                if (!cloud) continue;
                outCloud->append(cloud, outCloud->size());
            }
            successFlag = true;
        } else if (entity.isKindOf(CV_TYPES::MESH)) {
            ccMesh* outMesh = ccHObjectCaster::ToMesh(&entity);
            for (unsigned i = 0; i < container->getChildrenNumber(); ++i) {
                ccMesh* mesh = ccHObjectCaster::ToMesh(container->getChild(i));
                if (!mesh) continue;

                if (!outMesh->merge(mesh, false)) {
                    cloudViewer::utility::LogWarning(
                            "[AutoReadEntity] merge mesh child failed!");
                }
            }
            successFlag = true;
        } else if (entity.isA(CV_TYPES::HIERARCHY_OBJECT)) {
            if (entity.getChildrenNumber() > 0) {
                successFlag = true;
            } else {
                successFlag = false;
            }
        }
    } else {
        successFlag = false;
    }

    container->removeAllChildren();
    return successFlag;
}

bool AutoWriteEntity(const std::string& filename,
                     const ccHObject& entity,
                     const WritePointCloudOption& params) {
    FileIOFilter::SaveParameters parameters;
    {
        parameters.alwaysDisplaySaveDialog = params.print_progress;
        parameters.parentWidget = nullptr;
    }

    if (FileIOFilter::GetFilters().empty()) {
        FileIOFilter::InitInternalFilters();
    }

    FileIOFilter::ResetSesionCounter();

    CC_FILE_ERROR result = CC_FERR_NO_ERROR;
    FileIOFilter::Shared filter(nullptr);

    // save file
    while (true) {
        // look for file extension (we trust Qt on this task)
        QString completeFileName = CVTools::ToQString(filename);

        // if the file name has no extension, we had a default one!

        if (QFileInfo(completeFileName).suffix().isEmpty())
            completeFileName +=
                    QString(".%1").arg(filter->getDefaultExtension());

        QString extension = QFileInfo(completeFileName).suffix();
        CC_FILE_ERROR result = CC_FERR_NO_ERROR;
        {
            // convert extension to file format
            filter = FileIOFilter::FindBestFilterForExtension(extension);

            // unknown extension?
            if (!filter) {
                cloudViewer::utility::LogWarning(
                        "[AutoWriteEntity] Can't guess file format: unhandled "
                        "file extension '%s'",
                        CVTools::FromQString(extension).c_str());
                result = CC_FERR_CONSOLE_ERROR;
                break;
            }

            try {
                result = filter->saveToFile(const_cast<ccHObject*>(&entity),
                                            completeFileName, parameters);
            } catch (...) {
                cloudViewer::utility::LogWarning(
                        "[AutoWriteEntity] CV has caught an unhandled "
                        "exception while saving file '%s'",
                        CVTools::FromQString(completeFileName).c_str());
                result = CC_FERR_CONSOLE_ERROR;
            }

            if (result != CC_FERR_NO_ERROR) {
                FileIOFilter::DisplayErrorMessage(result, "saving",
                                                  completeFileName);
            }
        }

        break;
    }

    return result == CC_FERR_NO_ERROR;
}

bool AutoReadMesh(const std::string& filename,
                  ccMesh& mesh,
                  const ReadTriangleMeshOptions& params /*={}*/) {
    ReadPointCloudOption p;
    p.print_progress = params.print_progress;
    return AutoReadEntity(filename, mesh, p);
}

bool AutoWriteMesh(const std::string& filename,
                   const ccMesh& mesh,
                   bool write_ascii /* = false*/,
                   bool compressed /* = false*/,
                   bool write_vertex_normals /* = true*/,
                   bool write_vertex_colors /* = true*/,
                   bool write_triangle_uvs /* = true*/,
                   bool print_progress) {
    WritePointCloudOption params;
    params.write_ascii = WritePointCloudOption::IsAscii(write_ascii);
    params.compressed = WritePointCloudOption::Compressed(compressed);
    params.print_progress = print_progress;
    return AutoWriteEntity(filename, mesh, params);
}

using namespace cloudViewer;
std::shared_ptr<ccHObject> CreateEntityFromFile(const std::string& filename,
                                                const std::string& format,
                                                bool print_progress) {
    auto entity = cloudViewer::make_shared<ccHObject>("Group");
    ReadEntity(filename, *entity, format, print_progress);
    return entity;
}

bool ReadEntity(const std::string& filename,
                ccHObject& entity,
                const std::string& format,
                bool print_progress) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }
    if (filename_ext.empty()) {
        utility::LogWarning("Read entity failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_entity_read_function.find(filename_ext);
    if (map_itr == file_extension_to_entity_read_function.end()) {
        utility::LogWarning("Read entity failed: unknown file extension.");
        return false;
    }

    ReadPointCloudOption p;
    p.format = format;
    utility::ConsoleProgressUpdater progress_updater(
            std::string("Reading ") + utility::ToUpper(filename_ext) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;

    bool success = map_itr->second(filename, entity, p);
    utility::LogDebug("[ReadEntity] load {:d} entities.",
                      (int)entity.getChildrenNumber());
    return success;
}

bool WriteEntity(const std::string& filename,
                 const ccHObject& entity,
                 bool write_ascii /* = false*/,
                 bool compressed /* = false*/,
                 bool print_progress) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning("Write entity failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_entity_write_function.find(filename_ext);
    if (map_itr == file_extension_to_entity_write_function.end()) {
        utility::LogWarning("Write entity failed: unknown file extension.");
        return false;
    }

    WritePointCloudOption p;
    p.write_ascii = WritePointCloudOption::IsAscii(write_ascii);
    p.compressed = WritePointCloudOption::Compressed(compressed);
    utility::ConsoleProgressUpdater progress_updater(
            std::string("Writing ") + utility::ToUpper(filename_ext) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;

    bool success = map_itr->second(filename, entity, p);
    utility::LogDebug("[WriteEntity] Write {:d} entities.",
                      (int)entity.getChildrenNumber());
    return success;
}

}  // namespace io
}  // namespace cloudViewer
