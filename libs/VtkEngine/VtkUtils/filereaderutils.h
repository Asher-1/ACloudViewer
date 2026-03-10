// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file filereaderutils.h
/// @brief VTK file readers for various mesh formats (STL, OBJ, PLY, NASTRAN,
/// etc.).

#include <vtkDataObject.h>
#include <vtkFLUENTReader.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkLSDynaReader.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkOBJReader.h>
#include <vtkPDBReader.h>
#include <vtkPLYReader.h>
#include <vtkSTLReader.h>

#include <QMap>
#include <QObject>
#include <QRunnable>

#include "qVTK.h"
#include "vtkutils.h"

class vtkDataArray;

namespace VtkUtils {

/// @class ReaderTempl
/// @brief Template base for VTK file readers; provides dataObject() and
/// reader() access.
/// @tparam T VTK reader type
/// @tparam P Output data object type (default vtkDataObject)
template <class T, class P = vtkDataObject>
class ReaderTempl {
public:
    /// @return Output data object from reader, or nullptr if not loaded
    virtual P* dataObject() const {
        if (m_reader) return m_reader->GetOutput();
        return nullptr;
    }

    /// @return Underlying VTK reader instance
    T* reader() const { return m_reader; }

protected:
    T* m_reader = nullptr;
    P* m_dataObject = nullptr;
};

/// @class AbstractFileReader
/// @brief Base class for VTK file readers; runs as QRunnable and emits
/// finished().
class QVTK_ENGINE_LIB_API AbstractFileReader : public QObject,
                                               public QRunnable {
    Q_OBJECT
public:
    explicit AbstractFileReader(QObject* parent = nullptr);

    /// @param file Path to file to read
    void setFileName(const QString& file);
    /// @return Current file path
    QString fileName() const;

    /// @param title Display title for the loaded data
    void setTitle(const QString& title);
    /// @return Display title
    QString title() const;

signals:
    void finished();

protected:
    QString m_fileName;
    QString m_title;
};

/// @class VtkFileReader
/// @brief Reads legacy VTK format files.
class QVTK_ENGINE_LIB_API VtkFileReader
    : public AbstractFileReader,
      public ReaderTempl<vtkGenericDataObjectReader> {
    Q_OBJECT
public:
    explicit VtkFileReader(QObject* parent = nullptr);

    void run();
};

/// @class StlFileReader
/// @brief Reads STL mesh files.
class QVTK_ENGINE_LIB_API StlFileReader : public AbstractFileReader,
                                          public ReaderTempl<vtkSTLReader> {
    Q_OBJECT
public:
    explicit StlFileReader(QObject* parent = nullptr);

    void run();
};

/// @class ObjFileReader
/// @brief Reads OBJ mesh files.
class QVTK_ENGINE_LIB_API ObjFileReader : public AbstractFileReader,
                                          public ReaderTempl<vtkOBJReader> {
    Q_OBJECT
public:
    explicit ObjFileReader(QObject* parent = nullptr);

    void run();
};

/// @class DynaFileReader
/// @brief Reads LS-DYNA format files.
class QVTK_ENGINE_LIB_API DynaFileReader
    : public AbstractFileReader,
      public ReaderTempl<vtkLSDynaReader, vtkMultiBlockDataSet> {
    Q_OBJECT
public:
    explicit DynaFileReader(QObject* parent = nullptr);

    void run();
};

/// @class NastranFileReader
/// @brief Reads NASTRAN bulk data format files.
class QVTK_ENGINE_LIB_API NastranFileReader
    : public AbstractFileReader,
      public ReaderTempl<vtkGenericDataObjectReader> {
    Q_OBJECT
public:
    explicit NastranFileReader(QObject* parent = nullptr);

    void run();

protected:
    std::vector<int> m_matList;
    std::map<int, int> m_uniqMatIds;
};

/// @class FluentFileReader
/// @brief Reads FLUENT CFD mesh files.
class QVTK_ENGINE_LIB_API FluentFileReader
    : public AbstractFileReader,
      public ReaderTempl<vtkFLUENTReader, vtkMultiBlockDataSet> {
    Q_OBJECT
public:
    explicit FluentFileReader(QObject* parent = nullptr);

    void run();

protected:
    QMap<QString, vtkDataArray*> m_dataMap;
};

/// @class AnsysFileReader
/// @brief Reads ANSYS mesh format files.
class QVTK_ENGINE_LIB_API AnsysFileReader
    : public AbstractFileReader,
      public ReaderTempl<vtkGenericDataObjectReader> {
    Q_OBJECT
public:
    explicit AnsysFileReader(QObject* parent = nullptr);

    void run();

protected:
    void interpretFormatString(char* line,
                               int& fieldStart,
                               int& fieldWidth,
                               int& expectedLineLength) const;
    void interpretFormatStringEx(char* line,
                                 int& firstFieldWidth,
                                 int& fieldStart,
                                 int& fieldWidth,
                                 int& expectedLineLength) const;
    void interpret(const char* fmt, int& fieldWidth, int& linelen) const;
};

/// @class PlyFileReader
/// @brief Reads PLY mesh files.
class QVTK_ENGINE_LIB_API PlyFileReader : public AbstractFileReader,
                                          public ReaderTempl<vtkPLYReader> {
    Q_OBJECT
public:
    explicit PlyFileReader(QObject* parent = nullptr);

    void run();
};

/// @class PdbFileReader
/// @brief Reads PDB molecular structure files.
class QVTK_ENGINE_LIB_API PdbFileReader : public AbstractFileReader,
                                          public ReaderTempl<vtkPDBReader> {
    Q_OBJECT
public:
    explicit PdbFileReader(QObject* parent = nullptr);

    void run();
};

}  // namespace VtkUtils
