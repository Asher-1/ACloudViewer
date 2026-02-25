// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CV_DB_LIB
#include <ecvHObject.h>

// Local
#include "ecvGlobalShiftManager.h"

class QWidget;
class ccHObject;

/**
 * @brief File I/O error codes
 *
 * Enumeration of possible errors that can occur during file I/O operations.
 */
enum CC_FILE_ERROR {
    CC_FERR_NO_ERROR,                            ///< No error
    CC_FERR_BAD_ARGUMENT,                        ///< Invalid argument
    CC_FERR_UNKNOWN_FILE,                        ///< Unknown file format
    CC_FERR_WRONG_FILE_TYPE,                     ///< Wrong file type
    CC_FERR_WRITING,                             ///< Error writing file
    CC_FERR_READING,                             ///< Error reading file
    CC_FERR_NO_SAVE,                             ///< Save not supported
    CC_FERR_NO_LOAD,                             ///< Load not supported
    CC_FERR_BAD_ENTITY_TYPE,                     ///< Unsupported entity type
    CC_FERR_CANCELED_BY_USER,                    ///< Operation canceled by user
    CC_FERR_NOT_ENOUGH_MEMORY,                   ///< Insufficient memory
    CC_FERR_MALFORMED_FILE,                      ///< Malformed file structure
    CC_FERR_CONSOLE_ERROR,                       ///< Console error
    CC_FERR_BROKEN_DEPENDENCY_ERROR,             ///< Broken dependency
    CC_FERR_FILE_WAS_WRITTEN_BY_UNKNOWN_PLUGIN,  ///< Unknown plugin file
    CC_FERR_THIRD_PARTY_LIB_FAILURE,    ///< Third-party library failure
    CC_FERR_THIRD_PARTY_LIB_EXCEPTION,  ///< Third-party library exception
    CC_FERR_NOT_IMPLEMENTED,            ///< Feature not implemented
    CC_FERR_INTERNAL,                   ///< Internal error
};

/**
 * @class FileIOFilter
 * @brief Generic file I/O filter base class
 *
 * Abstract base class providing a common interface for file import/export
 * filters. Specific file format handlers must inherit from this class and
 * implement the virtual methods for loading and saving.
 */
class FileIOFilter {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~FileIOFilter() = default;

    /**
     * @struct LoadParameters
     * @brief Parameters for loading files
     *
     * Contains settings for coordinate shift handling, normal computation,
     * and dialog display during file loading.
     */
    struct LoadParameters {
        /**
         * @brief Default constructor
         *
         * Initializes all parameters to their default values.
         */
        LoadParameters()
            : shiftHandlingMode(ecvGlobalShiftManager::DIALOG_IF_NECESSARY),
              alwaysDisplayLoadDialog(true),
              coordinatesShiftEnabled(nullptr),
              coordinatesShift(nullptr),
              preserveShiftOnSave(true),
              autoComputeNormals(false),
              parentWidget(nullptr),
              sessionStart(true) {}

        ecvGlobalShiftManager::Mode
                shiftHandlingMode;      ///< How to handle large coordinates
        bool alwaysDisplayLoadDialog;   ///< Always display load dialog
        bool* coordinatesShiftEnabled;  ///< Output: whether shift was applied
        CCVector3d* coordinatesShift;   ///< Output: applied coordinate shift
        bool preserveShiftOnSave;       ///< Preserve shift when saving
        bool autoComputeNormals;        ///< Auto-compute normals if possible
        QWidget* parentWidget;          ///< Parent widget for dialogs
        bool sessionStart;  ///< Whether this is the first load of a session
    };

    /**
     * @struct SaveParameters
     * @brief Parameters for saving files
     *
     * Contains settings for dialog display during file saving.
     */
    struct SaveParameters {
        /**
         * @brief Default constructor
         */
        SaveParameters()
            : alwaysDisplaySaveDialog(true), parentWidget(nullptr) {}

        bool alwaysDisplaySaveDialog;  ///< Always display save dialog
        QWidget* parentWidget;         ///< Parent widget for dialogs
    };

    /**
     * @brief Shared pointer type
     */
    using Shared = QSharedPointer<FileIOFilter>;

public:  // public interface
    /**
     * @brief Check if import is supported
     * @return true if this filter can import files
     */
    CV_IO_LIB_API bool importSupported() const;

    /**
     * @brief Check if export is supported
     * @return true if this filter can export files
     */
    CV_IO_LIB_API bool exportSupported() const;

    /**
     * @brief Get file filter strings
     *
     * Returns filter strings for file dialogs, e.g., "ASCII file (*.asc)".
     * @param onImport true for import filters, false for export filters
     * @return List of filter strings
     */
    CV_IO_LIB_API const QStringList& getFileFilters(bool onImport) const;

    /**
     * @brief Get default file extension
     * @return Default file extension (without dot)
     */
    CV_IO_LIB_API QString getDefaultExtension() const;

public:  // public interface (to be reimplemented by each I/O filter)
    /**
     * @brief Load entities from a file
     *
     * This method must be implemented by derived classes.
     * @param filename File path to load from
     * @param container Container to store loaded entities
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) {
        Q_UNUSED(filename);
        Q_UNUSED(container);
        Q_UNUSED(parameters);

        return CC_FERR_NOT_IMPLEMENTED;
    }

    /**
     * @brief Save entities to a file
     *
     * This method must be implemented by derived classes.
     * @param entity Entity or group of entities to save
     * @param filename Output file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) {
        Q_UNUSED(entity);
        Q_UNUSED(filename);
        Q_UNUSED(parameters);

        return CC_FERR_NOT_IMPLEMENTED;
    }

    /**
     * @brief Check if entity type can be saved
     * @param type Entity type
     * @param[out] multiple Whether multiple instances can be saved at once
     * @param[out] exclusive Whether only this type can be saved (no mixing)
     * @return true if the entity type can be saved
     */
    virtual bool canSave(CV_CLASS_ENUM type,
                         bool& multiple,
                         bool& exclusive) const {
        Q_UNUSED(type);
        Q_UNUSED(multiple);
        Q_UNUSED(exclusive);

        return false;
    }

public:  // static methods
    /**
     * @brief Get list of all available import filters
     *
     * Returns a list of filter strings for use in file dialogs.
     * Includes "All (*)" as the first item.
     * @return List of import filter strings
     */
    CV_IO_LIB_API static QStringList ImportFilterList();

    /**
     * @brief Load file using a specific filter
     *
     * Convenience method to load entities from a file using a known filter.
     * @param filename File path to load
     * @param parameters Loading parameters
     * @param filter I/O filter to use
     * @param[out] result Error code
     * @return Loaded entities (nullptr on error)
     */
    CV_IO_LIB_API static ccHObject* LoadFromFile(const QString& filename,
                                                 LoadParameters& parameters,
                                                 Shared filter,
                                                 CC_FILE_ERROR& result);

    /**
     * @brief Load file with automatic filter selection
     *
     * Convenience method to load entities from a file. If fileFilter is empty,
     * the best filter is automatically determined from the file extension.
     * @param filename File path to load
     * @param parameters Loading parameters
     * @param[out] result Error code
     * @param fileFilter Filter string (empty for auto-detection)
     * @return Loaded entities (nullptr on error)
     */
    CV_IO_LIB_API static ccHObject* LoadFromFile(
            const QString& filename,
            LoadParameters& parameters,
            CC_FILE_ERROR& result,
            const QString& fileFilter = QString());

    //! Saves an entity (or a group of) to a specific file thanks to a given
    //! filter
    /** Shortcut to FileIOFilter::saveFile
            \param entities entity to save (can be a group of other entities)
            \param filename filename
            \param parameters saving parameters
            \param filter output filter
            \return error type (if any)
    **/
    CV_IO_LIB_API static CC_FILE_ERROR SaveToFile(
            ccHObject* entities,
            const QString& filename,
            const SaveParameters& parameters,
            Shared filter);

    //! Saves an entity (or a group of) to a specific file thanks to a given
    //! filter
    /** Shortcut to the other version of FileIOFilter::SaveToFile
            \param entities entity to save (can be a group of other entities)
            \param filename filename
            \param parameters saving parameters
            \param fileFilter output filter 'file filter'
            \return error type (if any)
    **/
    CV_IO_LIB_API static CC_FILE_ERROR SaveToFile(
            ccHObject* entities,
            const QString& filename,
            const SaveParameters& parameters,
            const QString& fileFilter);

    //! Shortcut to the ecvGlobalShiftManager mechanism specific for files
    /** \param[in] P sample point (typically the first loaded)
            \param[out] Pshift global shift
            \param[out] preserveCoordinateShift whether shift sould be preserved
    on save \param[in] loadParameters loading parameters \param[in]
    useInputCoordinatesShiftIfPossible whether to use the input 'PShift' vector
    if possible \return whether global shift has been defined/enabled
    **/
    CV_IO_LIB_API static bool HandleGlobalShift(
            const CCVector3d& P,
            CCVector3d& Pshift,
            bool& preserveCoordinateShift,
            LoadParameters& loadParameters,
            bool useInputCoordinatesShiftIfPossible = false);

    //! Displays (to console) the message corresponding to a given error code
    /** \param err error code
            \param action "saving", "reading", etc.
            \param filename corresponding file
    **/
    CV_IO_LIB_API static void DisplayErrorMessage(CC_FILE_ERROR err,
                                                  const QString& action,
                                                  const QString& filename);

    //! Returns whether special characters are present in the input string
    CV_IO_LIB_API static bool CheckForSpecialChars(const QString& filename);

public:  // loading "sessions" management
    //! Indicates to the I/O filters that a new loading/saving session has
    //! started (for "Apply all" buttons for instance)
    CV_IO_LIB_API static void ResetSesionCounter();

    //! Indicates to the I/O filters that a new loading/saving action has
    //! started
    /** \return the updated session counter
     **/
    CV_IO_LIB_API static unsigned IncreaseSesionCounter();

public:  // global filters registration mechanism
    //! Init internal filters (should be called once)
    CV_IO_LIB_API static void InitInternalFilters();

    //! Registers a new filter
    CV_IO_LIB_API static void Register(Shared filter);

    //! Returns the filter corresponding to the given 'file filter'
    CV_IO_LIB_API static Shared GetFilter(const QString& fileFilter,
                                          bool onImport);

    //! Returns the best filter (presumably) to open a given file extension
    CV_IO_LIB_API static Shared FindBestFilterForExtension(const QString& ext);

    //! Type of a I/O filters container
    using FilterContainer = std::vector<FileIOFilter::Shared>;

    //! Returns the set of all registered filters
    CV_IO_LIB_API static const FilterContainer& GetFilters();

    //! Unregisters all filters
    /** Should be called at the end of the application
     **/
    CV_IO_LIB_API static void UnregisterAll();

    //! Called when the filter is unregistered
    /** Does nothing by default **/
    virtual void unregister() {}

public:
    enum FilterFeature {
        NoFeatures = 0x0000,

        Import = 0x00001,  //< Imports data
        Export = 0x0002,   //< Exports data

        BuiltIn = 0x0004,  //< Implemented in the core

        DynamicInfo = 0x0008,  //< FilterInfo cannot be set statically (this is
                               // used for internal consistency checking)
    };
    Q_DECLARE_FLAGS(FilterFeatures, FilterFeature)

protected:
    static constexpr float DEFAULT_PRIORITY = 25.0f;

    struct FilterInfo {
        //! ID used to uniquely identify the filter (not user-visible)
        QString id;

        //! Priority used to determine sort order and which one is the default
        //! in the case of multiple FileIOFilters registering the same
        //! extension. Default is 25.0 /see DEFAULT_PRIORITY.
        float priority;

        //! List of extensions this filter can read (lowercase)
        //! e.g. "txt", "foo", "bin"
        //! This is used in FindBestFilterForExtension()
        QStringList importExtensions;

        //! The default file extension (for export)
        QString defaultExtension;

        //! List of file filters for import (e.g. "Test (*.txt)", "Foo (*.foo))
        QStringList importFileFilterStrings;

        //! List of file filters for export (e.g. "Test (*.txt)", "Foo (*.foo))
        QStringList exportFileFilterStrings;

        //! Supported features \see FilterFeature
        FilterFeatures features;
    };

    CV_IO_LIB_API explicit FileIOFilter(const FilterInfo& info);

    //! Allow import extensions to be set after construction
    //! (e.g. for ImageFileFilter & QImageReader::supportedImageFormats())
    void setImportExtensions(const QStringList& extensions);

    //! Allow import filter strings to be set after construction
    //! (e.g. for ImageFileFilter & QImageReader::supportedImageFormats())
    void setImportFileFilterStrings(const QStringList& filterStrings);

    //! Allow export filter strings to be set after construction
    //! (e.g. for ImageFileFilter & QImageReader::supportedImageFormats())
    void setExportFileFilterStrings(const QStringList& filterStrings);

private:
    void checkFilterInfo() const;

    FilterInfo m_filterInfo;
};

Q_DECLARE_OPERATORS_FOR_FLAGS(FileIOFilter::FilterFeatures)
