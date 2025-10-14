// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_ORDER_CHOICE_DIALOG_HEADER
#define ECV_ORDER_CHOICE_DIALOG_HEADER

// Qt
#include <QDialog>

class ccHObject;
class ecvMainAppInterface;
class Ui_RoleChoiceDialog;

//! Dialog to assign roles to two entities (e.g. compared/reference)
class ccOrderChoiceDlg : public QDialog {
    Q_OBJECT

public:
    //! Default constructor
    ccOrderChoiceDlg(ccHObject* firstEntity,
                     QString firstRole,
                     ccHObject* secondEntity,
                     QString secondRole,
                     ecvMainAppInterface* app = 0);

    //! Destructor
    virtual ~ccOrderChoiceDlg();

    //! Returns the first entity (new order)
    ccHObject* getFirstEntity();
    //! Returns the second entity (new order)
    ccHObject* getSecondEntity();

protected slots:

    //! Swaps the entities
    void swap();

protected:
    //! Sets the right colors to the entities and updates the dialog
    void setColorsAndLabels();

    Ui_RoleChoiceDialog* m_gui;
    ecvMainAppInterface* m_app;
    ccHObject* m_firstEnt;
    ccHObject* m_secondEnt;
    bool m_useInputOrder;
};

#endif  // ECV_ORDER_CHOICE_DIALOG_HEADER
