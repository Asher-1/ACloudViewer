// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_PRIMITIVE_DISTANCE_DIALOG_HEADER
#define ECV_PRIMITIVE_DISTANCE_DIALOG_HEADER

// Qt
#include <ui_primitiveDistanceDlg.h>

#include <QDialog>
#include <QString>

class ccHObject;
class ccPointCloud;
class ccGenericPointCloud;
class ccGenericMesh;

//! Dialog for cloud sphere or cloud plane comparison setting
class ecvPrimitiveDistanceDlg : public QDialog,
                                public Ui::primitiveDistanceDlg {
    Q_OBJECT

public:
    //! Default constructor
    ecvPrimitiveDistanceDlg(QWidget* parent = nullptr);

    //! Default destructor
    ~ecvPrimitiveDistanceDlg() = default;

    bool signedDistances() { return signedDistCheckBox->isChecked(); }
    bool flipNormals() { return flipNormalsCheckBox->isChecked(); }
    bool treatPlanesAsBounded() {
        return treatPlanesAsBoundedCheckBox->isChecked();
    }
public slots:
    void applyAndExit();
    void cancelAndExit();

protected slots:
    void toggleSigned(bool);
};

#endif  // ECV_PRIMITIVE_DISTANCE_DIALOG_HEADER
