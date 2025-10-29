// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CVPluginAPI.h"

// Qt
#include <QFrame>

// ECV_DB_LIB
#include <ecvColorScale.h>

class QComboBox;
class QToolButton;
class ccColorScalesManager;

//! Advanced editor for color scales
/** Combo-box + shortcut to color scale editor
 **/
class CVPLUGIN_LIB_API ccColorScaleSelector : public QFrame {
    Q_OBJECT

public:
    //! Default constructor
    ccColorScaleSelector(ccColorScalesManager* manager,
                         QWidget* parent,
                         QString defaultButtonIconPath = QString());

    //! Inits selector with the Color Scales Manager
    void init();

    //! Sets selected combo box item (scale) by UUID
    void setSelectedScale(QString uuid);

    //! Returns currently selected color scale
    ccColorScale::Shared getSelectedScale() const;

    //! Returns a given color scale by index
    ccColorScale::Shared getScale(int index) const;

signals:

    //! Signal emitted when a color scale is selected
    void colorScaleSelected(int);

    //! Signal emitted when the user clicks on the 'Spawn Color scale editor'
    //! button
    void colorScaleEditorSummoned();

protected:
    //! Color scales manager
    ccColorScalesManager* m_manager;

    //! Color scales combo-box
    QComboBox* m_comboBox;

    //! Spawn color scale editor button
    QToolButton* m_button;
};