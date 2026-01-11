// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "qPCL.h"

// QT
#include <QtCore/QHashFunctions>
#include <QtCore/QSharedPointer>
#include <QtCore/QString>
#include <QtCore/QVector>

// VTK
#include <vtkSmartPointer.h>

// Forward declarations (VTK types - not exposed in public API)
class vtkIdTypeArray;
class vtkPolyData;
class vtkActor;

//=============================================================================
// Selection Types and Enums (merged from cvSelectionTypes.h)
//=============================================================================

/**
 * @brief Selection modes matching ParaView's SelectionMode enum
 *
 * Reference: pqRenderViewSelectionReaction.h, lines 36-58
 */
enum class SelectionMode {
    // Surface selection
    SELECT_SURFACE_CELLS,           ///< Select cells on surface (rectangle)
    SELECT_SURFACE_POINTS,          ///< Select points on surface (rectangle)
    SELECT_SURFACE_CELLS_POLYGON,   ///< Select cells on surface (polygon)
    SELECT_SURFACE_POINTS_POLYGON,  ///< Select points on surface (polygon)

    // Frustum selection
    SELECT_FRUSTUM_CELLS,   ///< Select cells in frustum
    SELECT_FRUSTUM_POINTS,  ///< Select points in frustum

    // Block selection
    SELECT_BLOCKS,          ///< Select blocks (rectangle)
    SELECT_FRUSTUM_BLOCKS,  ///< Select blocks in frustum

    // Interactive selection
    SELECT_SURFACE_CELLS_INTERACTIVELY,      ///< Hover highlight + click select
                                             ///< (cells)
    SELECT_SURFACE_POINTS_INTERACTIVELY,     ///< Hover highlight + click select
                                             ///< (points)
    SELECT_SURFACE_CELLDATA_INTERACTIVELY,   ///< Hover highlight cell data
    SELECT_SURFACE_POINTDATA_INTERACTIVELY,  ///< Hover highlight point data

    // Tooltip selection (Hovering mode)
    HOVER_CELLS_TOOLTIP,   ///< Show cell data tooltip on hover (read-only)
    HOVER_POINTS_TOOLTIP,  ///< Show point data tooltip on hover (read-only)

    // Custom selection (emit signals with selection region)
    SELECT_CUSTOM_POLYGON,  ///< Custom polygon selection (signal only)

    // Zoom mode (ParaView-aligned)
    ZOOM_TO_BOX,  ///< Zoom to box region

    // Selection management
    CLEAR_SELECTION,  ///< Clear current selection
    GROW_SELECTION,   ///< Expand selection by one layer
    SHRINK_SELECTION  ///< Shrink selection by one layer
};

/**
 * @brief Selection modifiers for combining selections
 *
 * Reference: pqView.h, PV_SELECTION_* enums
 */
enum class SelectionModifier {
    SELECTION_DEFAULT,      ///< Replace selection (default)
    SELECTION_ADDITION,     ///< Add to selection (Ctrl)
    SELECTION_SUBTRACTION,  ///< Subtract from selection (Shift)
    SELECTION_TOGGLE        ///< Toggle selection (Ctrl+Shift)
};

// Provide qHash function for SelectionMode to support QSet<SelectionMode>
inline uint qHash(SelectionMode key, uint seed = 0) noexcept {
    return qHash(static_cast<int>(key), seed);
}

// Provide qHash function for SelectionModifier to support
// QSet<SelectionModifier>
inline uint qHash(SelectionModifier key, uint seed = 0) noexcept {
    return qHash(static_cast<int>(key), seed);
}

//=============================================================================
// Selection Data Classes
//=============================================================================

/**
 * @brief Information about a selected actor/representation
 *
 * Encapsulates per-actor selection information, similar to ParaView's
 * approach of maintaining separate collections for representations and
 * selection sources.
 */
struct QPCL_ENGINE_LIB_API cvActorSelectionInfo {
    vtkActor* actor;        ///< The selected actor (weak pointer)
    vtkPolyData* polyData;  ///< The associated polyData (weak pointer)
    double zValue;  ///< Z-buffer depth value (for front-to-back ordering)
    int propId;     ///< Unique prop ID (from hardware selector)
    unsigned int blockIndex;  ///< Block index for composite datasets
    QString actorName;        ///< Optional actor name/description

    cvActorSelectionInfo()
        : actor(nullptr),
          polyData(nullptr),
          zValue(1.0),
          propId(-1),
          blockIndex(0),
          actorName("") {}
};

/**
 * @brief Encapsulates selection data without exposing VTK types
 *
 * This class provides a VTK-independent interface for selection data,
 * hiding the underlying vtkIdTypeArray implementation.
 *
 * Following ParaView's design pattern, this class can now store information
 * about multiple selected actors/representations along with their selection
 * data.
 */
class QPCL_ENGINE_LIB_API cvSelectionData {
public:
    /**
     * @brief Field association for selection
     */
    enum FieldAssociation {
        CELLS = 0,  ///< Selection applies to cells
        POINTS = 1  ///< Selection applies to points
    };

    /**
     * @brief Construct empty selection data
     */
    cvSelectionData();

    /**
     * @brief Construct from VTK selection array
     * @param vtkArray The VTK array (will be deep copied)
     * @param association Field association (0=cells, 1=points)
     */
    cvSelectionData(vtkIdTypeArray* vtkArray, int association);

    /**
     * @brief Construct from ID vector
     * @param ids Vector of selected IDs
     * @param association Field association
     */
    cvSelectionData(const QVector<qint64>& ids, FieldAssociation association);

    /**
     * @brief Copy constructor
     */
    cvSelectionData(const cvSelectionData& other);

    /**
     * @brief Assignment operator
     */
    cvSelectionData& operator=(const cvSelectionData& other);

    /**
     * @brief Destructor
     */
    ~cvSelectionData();

    /**
     * @brief Check if selection is empty
     */
    bool isEmpty() const;

    /**
     * @brief Get number of selected items
     */
    int count() const;

    /**
     * @brief Get field association
     */
    FieldAssociation fieldAssociation() const { return m_fieldAssociation; }

    /**
     * @brief Get selected IDs as a vector (copy)
     */
    QVector<qint64> ids() const;

    /**
     * @brief Get the underlying VTK array (for internal use only)
     * @return Smart pointer to VTK array
     */
    vtkSmartPointer<vtkIdTypeArray> vtkArray() const { return m_vtkArray; }

    /**
     * @brief Clear the selection
     */
    void clear();

    /**
     * @brief Get human-readable field type string
     */
    QString fieldTypeString() const;

    ///@{
    /**
     * @brief Actor/Representation information (ParaView-style)
     *
     * These methods provide access to information about which actors/
     * representations were involved in the selection, similar to ParaView's
     * selectedRepresentations collection.
     */

    /**
     * @brief Add actor information to the selection
     * @param info Actor selection information
     *
     * Use this when performing hardware selection or when you need to track
     * which specific actor was selected.
     */
    void addActorInfo(const cvActorSelectionInfo& info);

    /**
     * @brief Set actor information (single actor case)
     * @param actor The selected actor
     * @param polyData The associated polyData
     * @param zValue Optional Z-buffer depth value
     */
    void setActorInfo(vtkActor* actor,
                      vtkPolyData* polyData,
                      double zValue = 1.0);

    /**
     * @brief Get number of actors in this selection
     * @return Number of actors (0 if no actor info stored)
     */
    int actorCount() const { return m_actorInfos.size(); }

    /**
     * @brief Check if actor information is available
     */
    bool hasActorInfo() const { return !m_actorInfos.isEmpty(); }

    /**
     * @brief Get actor info at index
     * @param index Index in the list (0 = front-most)
     * @return Actor info, or empty struct if index out of range
     *
     * Note: List is sorted by Z-value (front to back)
     */
    cvActorSelectionInfo actorInfo(int index = 0) const;

    /**
     * @brief Get all actor infos
     * @return Vector of actor infos, sorted by Z-value (front to back)
     */
    QVector<cvActorSelectionInfo> actorInfos() const { return m_actorInfos; }

    /**
     * @brief Get the primary (front-most) actor
     * @return Pointer to primary actor, or nullptr if not available
     *
     * This is the actor closest to the camera (smallest Z value)
     */
    vtkActor* primaryActor() const;

    /**
     * @brief Get the primary (front-most) polyData
     * @return Pointer to primary polyData, or nullptr if not available
     */
    vtkPolyData* primaryPolyData() const;

    /**
     * @brief Clear actor information
     */
    void clearActorInfo();

    ///@}

private:
    vtkSmartPointer<vtkIdTypeArray>
            m_vtkArray;  ///< Internal VTK array (smart pointer managed)
    FieldAssociation m_fieldAssociation;  ///< Field association

    /// Actor/representation information (ParaView-style)
    /// Sorted by Z-value (front to back)
    QVector<cvActorSelectionInfo> m_actorInfos;
};

/**
 * @brief Shared pointer to selection data
 */
using cvSelectionDataPtr = QSharedPointer<cvSelectionData>;
