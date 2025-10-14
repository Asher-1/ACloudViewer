// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../casters.h"

#include <FileIO.h>
#include <FileIOFilter.h>

#include <QFileInfo>
#include <QWidget>

namespace py = pybind11;
using namespace pybind11::literals;
static void ThrowForFileError(CC_FILE_ERROR error)
{
    switch (error)
    {
    case CC_FERR_NO_ERROR:
        break;
    case CC_FERR_BAD_ARGUMENT:
        throw std::runtime_error("Bad argument");
    case CC_FERR_UNKNOWN_FILE:
        throw std::runtime_error("Unknown file");
    case CC_FERR_WRONG_FILE_TYPE:
        throw std::runtime_error("Wrong file type");
    case CC_FERR_WRITING:
        throw std::runtime_error("Error when writing");
    case CC_FERR_READING:
        throw std::runtime_error("Error when reading");
    case CC_FERR_NO_SAVE:
        throw std::runtime_error("Nothing to save");
    case CC_FERR_NO_LOAD:
        throw std::runtime_error("Nothing to load");
    case CC_FERR_BAD_ENTITY_TYPE:
        throw std::runtime_error("Bad entity type");
    case CC_FERR_CANCELED_BY_USER:
        throw std::runtime_error("Canceled by user");
    case CC_FERR_NOT_ENOUGH_MEMORY:
        throw std::runtime_error("Not enough memory");
    case CC_FERR_MALFORMED_FILE:
        throw std::runtime_error("Malformed File");
    case CC_FERR_CONSOLE_ERROR:
        throw std::runtime_error("The error has been logged in the console");
    case CC_FERR_BROKEN_DEPENDENCY_ERROR:
        throw std::runtime_error("Broken dependency");
    case CC_FERR_FILE_WAS_WRITTEN_BY_UNKNOWN_PLUGIN:
        throw std::runtime_error("File was written by unknown plugin");
    case CC_FERR_THIRD_PARTY_LIB_FAILURE:
        throw std::runtime_error("Third party lib failure");
    case CC_FERR_THIRD_PARTY_LIB_EXCEPTION:
        throw std::runtime_error("Third party lib exception");
    case CC_FERR_INTERNAL:
        throw std::runtime_error("Internal error");
    case CC_FERR_NOT_IMPLEMENTED:
        throw std::runtime_error("Not implemented");
    }
}

void define_qcc_io(py::module &m)
{
    py::class_<ecvGlobalShiftManager> PyccGlobalShiftManager(m, "ccGlobalShiftManager");

    py::native_enum<ecvGlobalShiftManager::Mode>(
        PyccGlobalShiftManager, "Mode", "enum.Enum", "ecvGlobalShiftManager::Mode.")
        .value("NO_DIALOG", ecvGlobalShiftManager::Mode::NO_DIALOG)
        .value("NO_DIALOG_AUTO_SHIFT", ecvGlobalShiftManager::Mode::NO_DIALOG_AUTO_SHIFT)
        .value("DIALOG_IF_NECESSARY", ecvGlobalShiftManager::Mode::DIALOG_IF_NECESSARY)
        .value("ALWAYS_DISPLAY_DIALOG", ecvGlobalShiftManager::Mode::ALWAYS_DISPLAY_DIALOG)
        .export_values()
        .finalize();

    py::class_<FileIOFilter> PyFileIOFilter(m, "FileIOFilter");
    PyFileIOFilter
        .def_static(
            "LoadFromFile",
            [](const QString &filename, FileIOFilter::LoadParameters &parameters)
            {
                CC_FILE_ERROR result = CC_FERR_NO_ERROR;
                ccHObject *newGroup = FileIOFilter::LoadFromFile(filename, parameters, result);
                ThrowForFileError(result);
                return newGroup;
            },
            py::return_value_policy::take_ownership)
        .def_static(
            "SaveToFile",
            [](ccHObject *entities,
               const QString &filename,
               const FileIOFilter::SaveParameters &parameters,
               const QString filterName = QString())
            {
                const QString requestedFilterName =
                    filterName.isEmpty() ? QFileInfo(filename).suffix() : filterName;

                const FileIOFilter::FilterContainer &availableFilters = FileIOFilter::GetFilters();
                for (const FileIOFilter::Shared &filter : availableFilters)
                {
                    const QStringList filters = filter->getFileFilters(false /* onImport */);
                    const auto it =
                        std::find_if(filters.begin(),
                                     filters.end(),
                                     [&requestedFilterName](const QString &filterString)
                                     { return filterString.contains(requestedFilterName); });
                    if (it != filters.end())
                    {
                        CC_FILE_ERROR error =
                            FileIOFilter::SaveToFile(entities, filename, parameters, filter);
                        ThrowForFileError(error);
                        return;
                    }
                }
                throw std::runtime_error(std::string("Unable to find FileFilter for ") +
                                         requestedFilterName.toStdString());
            },
            "entities"_a,
            "filename"_a,
            "parameters"_a,
            "fileFileter"_a = QString());

    py::class_<FileIOFilter::LoadParameters>(PyFileIOFilter, "LoadParameters")
        .def(py::init<>())
        .def_readwrite("shiftHandlingMode", &FileIOFilter::LoadParameters::shiftHandlingMode)
        .def_readwrite("alwaysDisplayLoadDialog",
                       &FileIOFilter::LoadParameters::alwaysDisplayLoadDialog)
        .def_readwrite("coordinatesShiftEnabled",
                       &FileIOFilter::LoadParameters::coordinatesShiftEnabled)
        .def_readwrite("coordinatesShift",
                       &FileIOFilter::LoadParameters::coordinatesShift,
                       py::return_value_policy::reference)
        .def_readwrite("preserveShiftOnSave", &FileIOFilter::LoadParameters::preserveShiftOnSave)
        .def_readwrite("autoComputeNormals", &FileIOFilter::LoadParameters::autoComputeNormals)
        .def_readwrite("parentWidget",
                       &FileIOFilter::LoadParameters::parentWidget,
                       py::return_value_policy::reference)
        .def_readwrite("sessionStart", &FileIOFilter::LoadParameters::sessionStart);

    py::class_<FileIOFilter::SaveParameters>(PyFileIOFilter, "SaveParameters")
        .def(py::init<>())
        .def_readwrite("alwaysDisplaySaveDialog",
                       &FileIOFilter::SaveParameters::alwaysDisplaySaveDialog)
        .def_readwrite("parentWidget",
                       &FileIOFilter::SaveParameters::parentWidget,
                       py::return_value_policy::reference);
}
