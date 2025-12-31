// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt - must be included before other headers for MOC to work correctly
#include <QtGlobal>

// clang-format off
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QAction>
#include <QCursor>
#include <QObject>
#include <QPointer>
#else
#include <QtWidgets/QAction>
#include <QtGui/QCursor>
#include <QtCore/QObject>
#include <QtCore/QPointer>
#endif
// clang-format on

#include "cvSelectionTypes.h"  // For SelectionMode and SelectionModifier enums
#include "cvViewSelectionManager.h"
#include "qPCL.h"

// Forward declarations (for types used in QPointer, need full definition)
// Note: cvRenderViewSelectionTool must be fully defined for QPointer template
// instantiation
#include "cvRenderViewSelectionTool.h"
class ecvGenericVisualizer3D;

/**
 * @brief cvSelectionReaction handles various selection modes available on
 * views.
 *
 * This class follows ParaView's pqRenderViewSelectionReaction pattern.
 * Create multiple instances of cvSelectionReaction to handle different
 * selection modes. The class uses internal static members to ensure that
 * at most 1 view (and 1 type of selection) is in selection-mode at any given
 * time.
 *
 * Reference: ParaView/Qt/ApplicationComponents/pqRenderViewSelectionReaction.h
 */
class QPCL_ENGINE_LIB_API cvSelectionReaction : public QObject {
    Q_OBJECT

public:
    using SelectionMode = ::SelectionMode;

    /**
     * @brief Constructor
     * @param parentAction The QAction that triggers this reaction
     * @param mode The selection mode for this reaction
     * @param modifierGroup Optional action group for selection modifiers
     */
    cvSelectionReaction(QAction* parentAction,
                        SelectionMode mode,
                        QActionGroup* modifierGroup = nullptr);

    ~cvSelectionReaction() override;

    /**
     * @brief Get the parent action
     */
    QAction* parentAction() const { return m_parentAction; }

    /**
     * @brief Get the selection mode
     */
    SelectionMode mode() const { return m_mode; }

    /**
     * @brief Set the visualizer for selection operations
     */
    void setVisualizer(ecvGenericVisualizer3D* viewer);

    /**
     * @brief Check if this reaction's selection is currently active
     */
    bool isActive() const;

    /**
     * @brief Get the currently active reaction (static)
     */
    static cvSelectionReaction* activeReaction() { return ActiveReaction; }

signals:
    /**
     * @brief Emitted when selection is finished
     */
    void selectionFinished(const cvSelectionData& selectionData);

    /**
     * @brief Emitted for custom box selection
     */
    void selectedCustomBox(int xmin, int ymin, int xmax, int ymax);

    /**
     * @brief Emitted for custom polygon selection
     */
    void selectedCustomPolygon(vtkIntArray* polygon);

    /**
     * @brief Emitted when zoom to box is requested
     */
    void zoomToBoxRequested(int xmin, int ymin, int xmax, int ymax);

public slots:
    /**
     * @brief Called when the action is triggered
     *
     * For checkable actions, calls beginSelection() or endSelection().
     * For non-checkable actions, calls both in sequence.
     */
    virtual void actionTriggered(bool val);

    /**
     * @brief Updates the enabled state of the action
     *
     * Handles enable state for CLEAR_SELECTION, GROW_SELECTION,
     * and SHRINK_SELECTION modes.
     */
    virtual void updateEnableState();

protected slots:
    /**
     * @brief Starts the selection mode
     */
    virtual void beginSelection();

    /**
     * @brief Ends the selection mode
     */
    virtual void endSelection();

    /**
     * @brief Handle selection completion from the tool
     */
    virtual void onToolSelectionFinished(const cvSelectionData& selectionData);

protected:
    /**
     * @brief Get the current selection modifier
     */
    int getSelectionModifier();

    /**
     * @brief Check if this selection is compatible with another mode
     */
    virtual bool isCompatible(SelectionMode mode);

    /**
     * @brief Get or create the selection tool for this mode
     */
    cvRenderViewSelectionTool* getOrCreateTool();

private:
    QPointer<QAction> m_parentAction;
    QPointer<QActionGroup> m_modifierGroup;
    SelectionMode m_mode;
    QPointer<cvRenderViewSelectionTool> m_tool;
    QCursor m_previousCursor;

    // Static: only one reaction can be active at a time
    static QPointer<cvSelectionReaction> ActiveReaction;
};
