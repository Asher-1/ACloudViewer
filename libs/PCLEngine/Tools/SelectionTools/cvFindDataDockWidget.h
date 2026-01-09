// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDockWidget>
#include <QPointer>
#include <QScrollArea>

#include "qPCL.h"

// Forward declarations
class cvSelectionPropertiesWidget;
class cvSelectionHighlighter;
class cvViewSelectionManager;
class cvSelectionData;
class ecvGenericVisualizer3D;
class ccHObject;

/**
 * @brief Dock widget for Find Data / Selection Properties panel.
 *
 * This widget provides a standalone dock panel for selection properties,
 * similar to ParaView's Find Data panel. It can be shown/hidden independently
 * of the selection tools state.
 *
 * Layout follows ParaView's pattern:
 * - Docked on the right side of the application
 * - Contains cvSelectionPropertiesWidget with collapsible sections
 * - Can be tabified with other dock widgets
 */
class QPCL_ENGINE_LIB_API cvFindDataDockWidget : public QDockWidget {
    Q_OBJECT

public:
    explicit cvFindDataDockWidget(QWidget* parent = nullptr);
    ~cvFindDataDockWidget() override;

    /**
     * @brief Get the selection properties widget contained in this dock.
     */
    cvSelectionPropertiesWidget* selectionPropertiesWidget() const {
        return m_selectionWidget;
    }

    /**
     * @brief Configure the dock with necessary components.
     * @param highlighter The selection highlighter for visual feedback
     * @param manager The selection manager for coordinating selections
     * @param visualizer The 3D visualizer for rendering
     */
    void configure(cvSelectionHighlighter* highlighter,
                   cvViewSelectionManager* manager,
                   ecvGenericVisualizer3D* visualizer);

    /**
     * @brief Update the selection display with new data.
     * @param selectionData The new selection data to display
     */
    void updateSelection(const cvSelectionData& selectionData);

    /**
     * @brief Clear the current selection display.
     */
    void clearSelection();

    /**
     * @brief Refresh the data producer list.
     * Call this when data sources change (e.g., after loading new data)
     */
    void refreshDataProducers();

signals:
    /**
     * @brief Emitted when an extracted object is ready to be added to the
     * scene.
     */
    void extractedObjectReady(ccHObject* obj);

    /**
     * @brief Emitted when the dock visibility changes.
     */
    void visibilityChanged(bool visible);

protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;

private:
    void setupUi();

private:
    QScrollArea* m_scrollArea;
    cvSelectionPropertiesWidget* m_selectionWidget;
};
