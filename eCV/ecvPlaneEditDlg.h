// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_PLANE_EDIT_DLG_HEADER
#define ECV_PLANE_EDIT_DLG_HEADER

// Local
#include <ui_planeEditDlg.h>

#include "ecvPickingListener.h"

// cloudViewer
#include <CVGeom.h>

// Qt
#include <QDialog>

class ccPlane;
class ccHObject;
class ccPickingHub;

//! Dialog to create (or edit the parameters) of a plane
class ccPlaneEditDlg : public QDialog,
                       public ccPickingListener,
                       public Ui::PlaneEditDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccPlaneEditDlg(ccPickingHub* pickingHub, QWidget* parent);

    //! Destructor
    virtual ~ccPlaneEditDlg();

    //! Links this dialog with an existing plane
    void initWithPlane(ccPlane* plane);

    //! Updates a plane with the current parameters
    void updatePlane(ccPlane* plane);

    //! Inherited from ccPickingListener
    virtual void onItemPicked(const PickedItem& pi);

public slots:

    void pickPointAsCenter(bool);
    void onDipDirChanged(double);
    void onDipDirModified(bool);
    void onNormalChanged(double);

protected slots:

    void saveParamsAndAccept();

protected:  // members
    //! Associated plane (if any)
    ccPlane* m_associatedPlane;

    //! Picking hub
    ccPickingHub* m_pickingHub;
};

#endif  // ECV_PLANE_EDIT_DLG_HEADER
