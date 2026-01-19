// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvFindDataDockWidget.h"

#include <CVLog.h>

#include <QShowEvent>
#include <QVBoxLayout>

#include "cvSelectionData.h"
#include "cvSelectionHighlighter.h"
#include "cvSelectionPropertiesWidget.h"
#include "cvViewSelectionManager.h"

//-----------------------------------------------------------------------------
cvFindDataDockWidget::cvFindDataDockWidget(QWidget* parent)
    : QDockWidget(parent), m_scrollArea(nullptr), m_selectionWidget(nullptr) {
    // Set dock properties - ParaView style
    setWindowTitle(tr("Find Data"));
    setObjectName("findDataDock");

    // Allow docking on left and right areas only (like ParaView)
    setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);

    // Set minimum width for readability
    setMinimumWidth(300);

    setupUi();

    CVLog::PrintVerbose("[cvFindDataDockWidget] Dock widget created");
}

//-----------------------------------------------------------------------------
cvFindDataDockWidget::~cvFindDataDockWidget() {
    CVLog::PrintVerbose("[cvFindDataDockWidget] Dock widget destroyed");
}

//-----------------------------------------------------------------------------
void cvFindDataDockWidget::setupUi() {
    // Create container widget
    QWidget* containerWidget = new QWidget(this);
    QVBoxLayout* containerLayout = new QVBoxLayout(containerWidget);
    containerLayout->setContentsMargins(0, 0, 0, 0);
    containerLayout->setSpacing(0);

    // Create scroll area - ParaView style
    m_scrollArea = new QScrollArea(containerWidget);
    m_scrollArea->setMinimumWidth(300);
    m_scrollArea->setFrameShape(QFrame::NoFrame);
    m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_scrollArea->setWidgetResizable(true);

    // Create selection properties widget
    m_selectionWidget = new cvSelectionPropertiesWidget(m_scrollArea);

    // Set the widget in scroll area
    m_scrollArea->setWidget(m_selectionWidget);

    // Add scroll area to container
    containerLayout->addWidget(m_scrollArea);

    // Set container as dock widget content
    setWidget(containerWidget);

    // Connect extractedObjectReady signal
    connect(m_selectionWidget,
            &cvSelectionPropertiesWidget::extractedObjectReady, this,
            &cvFindDataDockWidget::extractedObjectReady);
}

//-----------------------------------------------------------------------------
void cvFindDataDockWidget::configure(cvSelectionHighlighter* highlighter,
                                     cvViewSelectionManager* manager,
                                     ecvGenericVisualizer3D* visualizer) {
    if (!m_selectionWidget) {
        CVLog::Warning(
                "[cvFindDataDockWidget] Cannot configure - widget not "
                "initialized");
        return;
    }

    // Configure the selection properties widget
    // IMPORTANT: Set visualizer FIRST because other setters may trigger
    // operations that need the visualizer (e.g., setSelectionManager calls
    // updateDataProducerCombo)
    if (visualizer) {
        m_selectionWidget->setVisualizer(visualizer);
    } else {
        CVLog::Warning("[cvFindDataDockWidget] Visualizer is nullptr!");
    }

    if (highlighter) {
        m_selectionWidget->setHighlighter(highlighter);
    } else {
        CVLog::Warning("[cvFindDataDockWidget] Highlighter is nullptr!");
    }

    if (manager) {
        m_selectionWidget->setSelectionManager(manager);
    } else {
        CVLog::Warning("[cvFindDataDockWidget] Selection manager is nullptr!");
    }
}

//-----------------------------------------------------------------------------
void cvFindDataDockWidget::updateSelection(
        const cvSelectionData& selectionData) {
    if (m_selectionWidget) {
        m_selectionWidget->updateSelection(selectionData);
    }
}

//-----------------------------------------------------------------------------
void cvFindDataDockWidget::clearSelection() {
    if (m_selectionWidget) {
        m_selectionWidget->clearSelection();
    }
}

//-----------------------------------------------------------------------------
void cvFindDataDockWidget::refreshDataProducers() {
    if (m_selectionWidget) {
        m_selectionWidget->refreshDataProducers();
    }
}

//-----------------------------------------------------------------------------
void cvFindDataDockWidget::showEvent(QShowEvent* event) {
    QDockWidget::showEvent(event);
    emit visibilityChanged(true);

    // Refresh data producers when dock is shown
    // This ensures the combo is updated after data is loaded
    if (m_selectionWidget) {
        m_selectionWidget->refreshDataProducers();
    }

    CVLog::PrintVerbose("[cvFindDataDockWidget] Dock shown");
}

//-----------------------------------------------------------------------------
void cvFindDataDockWidget::hideEvent(QHideEvent* event) {
    QDockWidget::hideEvent(event);
    emit visibilityChanged(false);
    CVLog::PrintVerbose("[cvFindDataDockWidget] Dock hidden");
}
