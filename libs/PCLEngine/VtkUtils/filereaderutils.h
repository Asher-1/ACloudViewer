// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file filereaderutils.h
 * @brief VTK file reader utilities for various 3D file formats
 * 
 * Provides Qt-based file readers for multiple 3D formats:
 * - VTK legacy format
 * - STL (stereolithography)
 * - OBJ (Wavefront)
 * - PLY (Polygon File Format)
 * - PDB (Protein Data Bank)
 * - LS-DYNA
 * - FLUENT
 * - ANSYS
 * - Nastran
 * 
 * All readers are based on Qt's QRunnable for asynchronous loading.
 */

#pragma once

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

#include "qPCL.h"
#include "vtkutils.h"

class vtkDataArray;

namespace VtkUtils {

/**
 * @class ReaderTempl
 * @brief Template base class for VTK file readers
 * @tparam T VTK reader type
 * @tparam P VTK data object type (default: vtkDataObject)
 * 
 * Provides common interface for accessing reader and data object.
 */
template <class T, class P = vtkDataObject>
class ReaderTempl {
public:
    /**
     * @brief Get output data object from reader
     * @return Pointer to VTK data object, or nullptr if reader not set
     */
    virtual P* dataObject() const {
        if (m_reader) return m_reader->GetOutput();
        return nullptr;
    }

    /**
     * @brief Get underlying VTK reader
     * @return Pointer to VTK reader instance
     */
    T* reader() const { return m_reader; }

protected:
    T* m_reader = nullptr;         ///< VTK reader instance
    P* m_dataObject = nullptr;     ///< Cached data object
};

/**
 * @class AbstractFileReader
 * @brief Abstract base class for asynchronous file readers
 * 
 * Provides common interface for file reading with Qt threading support.
 * Subclasses implement specific format readers using VTK.
 */
class QPCL_ENGINE_LIB_API AbstractFileReader : public QObject,
                                               public QRunnable {
    Q_OBJECT
public:
    /**
     * @brief Constructor
     * @param parent Parent QObject (optional)
     */
    explicit AbstractFileReader(QObject* parent = nullptr);

    /**
     * @brief Set input file name
     * @param file Path to file to read
     */
    void setFileName(const QString& file);
    
    /**
     * @brief Get input file name
     * @return Current file path
     */
    QString fileName() const;

    /**
     * @brief Set reader title/description
     * @param title Reader title
     */
    void setTitle(const QString& title);
    
    /**
     * @brief Get reader title
     * @return Reader title
     */
    QString title() const;

signals:
    /**
     * @brief Emitted when file reading is complete
     */
    void finished();

protected:
    QString m_fileName;  ///< Input file path
    QString m_title;     ///< Reader title/description
};

/**
 * @class VtkFileReader
 * @brief Reader for VTK legacy format files
 * 
 * Reads VTK's legacy ASCII/binary file format (.vtk extension).
 */
class QPCL_ENGINE_LIB_API VtkFileReader
    : public AbstractFileReader,
      public ReaderTempl<vtkGenericDataObjectReader> {
    Q_OBJECT
public:
    /**
     * @brief Constructor
     * @param parent Parent QObject (optional)
     */
    explicit VtkFileReader(QObject* parent = nullptr);

    /**
     * @brief Execute file reading
     * 
     * Reads VTK file asynchronously. Emits finished() signal when complete.
     */
    void run();
};

/**
 * @class StlFileReader
 * @brief Reader for STL (stereolithography) files
 * 
 * Reads STL ASCII and binary formats (.stl extension).
 */
class QPCL_ENGINE_LIB_API StlFileReader : public AbstractFileReader,
                                          public ReaderTempl<vtkSTLReader> {
    Q_OBJECT
public:
    /**
     * @brief Constructor
     * @param parent Parent QObject (optional)
     */
    explicit StlFileReader(QObject* parent = nullptr);

    /**
     * @brief Execute file reading
     * 
     * Reads STL file asynchronously. Emits finished() signal when complete.
     */
    void run();
};

/**
 * @class ObjFileReader
 * @brief Reader for Wavefront OBJ files
 * 
 * Reads Wavefront OBJ format (.obj extension) with basic material support.
 */
class QPCL_ENGINE_LIB_API ObjFileReader : public AbstractFileReader,
                                          public ReaderTempl<vtkOBJReader> {
    Q_OBJECT
public:
    /**
     * @brief Constructor
     * @param parent Parent QObject (optional)
     */
    explicit ObjFileReader(QObject* parent = nullptr);

    /**
     * @brief Execute file reading
     * 
     * Reads OBJ file asynchronously. Emits finished() signal when complete.
     */
    void run();
};

class QPCL_ENGINE_LIB_API DynaFileReader
    : public AbstractFileReader,
      public ReaderTempl<vtkLSDynaReader, vtkMultiBlockDataSet> {
    Q_OBJECT
public:
    explicit DynaFileReader(QObject* parent = nullptr);

    void run();
};

class QPCL_ENGINE_LIB_API NastranFileReader
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

class QPCL_ENGINE_LIB_API FluentFileReader
    : public AbstractFileReader,
      public ReaderTempl<vtkFLUENTReader, vtkMultiBlockDataSet> {
    Q_OBJECT
public:
    explicit FluentFileReader(QObject* parent = nullptr);

    void run();

protected:
    QMap<QString, vtkDataArray*> m_dataMap;
};

class QPCL_ENGINE_LIB_API AnsysFileReader
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

class QPCL_ENGINE_LIB_API PlyFileReader : public AbstractFileReader,
                                          public ReaderTempl<vtkPLYReader> {
    Q_OBJECT
public:
    explicit PlyFileReader(QObject* parent = nullptr);

    void run();
};

class QPCL_ENGINE_LIB_API PdbFileReader : public AbstractFileReader,
                                          public ReaderTempl<vtkPDBReader> {
    Q_OBJECT
public:
    explicit PdbFileReader(QObject* parent = nullptr);

    void run();
};

}  // namespace VtkUtils
