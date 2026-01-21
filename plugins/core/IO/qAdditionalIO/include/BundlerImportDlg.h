// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

// local
#include "CV_io.h"
#include "ui_openBundlerFileDlg.h"

class ccGLMatrix;

//! Dialog for importation of Snavely's Bundler files
class /*CV_IO_LIB_API*/ BundlerImportDlg : public QDialog,
                                           public Ui::BundlerImportDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit BundlerImportDlg(QWidget* parent = 0);

    //! Destructor
    virtual ~BundlerImportDlg();

    //! Returns whether keypoints should be imported
    bool importKeypoints() const;
    //! Returns whether alternative keypoints should be used
    bool useAlternativeKeypoints() const;
    //! Returns whether images should be imported
    bool importImages() const;
    //! Returns whether images should be undistorted
    bool undistortImages() const;
    //! Returns whether images should be ortho-rectified as clouds
    bool orthoRectifyImagesAsClouds() const;
    //! Returns whether images should be ortho-rectified as images
    bool orthoRectifyImagesAsImages() const;
    //! Returns whether colored pseudo-DTM should be generated
    bool generateColoredDTM() const;
    //! Returns images should be kept in memory or not
    bool keepImagesInMemory() const;

    //! Image ortho-rectification methods
    enum OrthoRectMethod { OPTIMIZED, DIRECT_UNDISTORTED, DIRECT };
    //! Returns the ortho-rectification method (for images)
    OrthoRectMethod getOrthorectificationMethod() const;

    //! Sets keypoints count on initialization
    void setKeypointsCount(unsigned count);
    //! Sets cameras count on initialization
    void setCamerasCount(unsigned count);
    //! Sets file version on initialization
    void setVer(unsigned majorVer, unsigned minorVer);

    //! Sets default image list filename (full path)
    void setImageListFilename(const QString& filename);
    //! Gets image list filename (full path)
    QString getImageListFilename() const;

    //! Sets default alternative keypoints filename (full path)
    void setAltKeypointsFilename(const QString& filename);
    //! Gets alternative keypoints filename (full path)
    QString getAltKeypointsFilename() const;

    //! Returns scale factor
    double getScaleFactor() const;

    //! Returns desired number of vertices for DTM
    unsigned getDTMVerticesCount() const;

    //! Returns the optional transformation matrix (if defined)
    bool getOptionalTransfoMatrix(ccGLMatrix& mat);

protected slots:
    void browseImageListFilename();
    void browseAltKeypointsFilename();
    void acceptAndSaveSettings();

protected:
    //! Inits dialog state from persistent settings
    void initFromPersistentSettings();

    //! Saves dialog state from persistent settings
    void saveToPersistentSettings();
};
