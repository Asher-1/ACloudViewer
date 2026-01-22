// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "CV_db.h"

// Qt
#include <QAtomicInt>
#include <QProgressDialog>
#include <QTimer>

// CV_CORE_LIB
#include <GenericProgressCallback.h>

//! Graphical progress indicator (thread-safe)
/** Implements the GenericProgressCallback interface, in order
        to be passed to the cloudViewer algorithms (check the
        cloudViewer documentation for more information about the
        inherited methods).
**/
class CV_DB_LIB_API ecvProgressDialog
    : public QProgressDialog,
      public cloudViewer::GenericProgressCallback {
    Q_OBJECT

public:
    //! Default constructor
    /** By default, a cancel button is always displayed on the
            progress interface. It is only possible to activate or
            deactivate this button. Sadly, the fact that this button is
            activated doesn't mean it will be possible to stop the ongoing
            process: it depends only on the client algorithm implementation.
            \param cancelButton activates or deactivates the cancel button
            \param parent parent widget
    **/
    ecvProgressDialog(bool cancelButton = false, QWidget* parent = 0);

    //! Destructor (virtual)
    virtual ~ecvProgressDialog() {}

    // inherited method
    virtual void update(float percent) override;
    inline virtual void setMethodTitle(const char* methodTitle) override {
        setMethodTitle(QString(methodTitle));
    }
    inline virtual void setInfo(const char* infoStr) override {
        setInfo(QString(infoStr));
    }
    inline virtual bool isCancelRequested() override { return wasCanceled(); }
    virtual void start() override;
    virtual void stop() override;

    //! setMethodTitle with a QString as argument
    virtual void setMethodTitle(QString methodTitle);
    //! setInfo with a QString as argument
    virtual void setInfo(QString infoStr);

protected Q_SLOTS:

    //! Refreshes the progress
    /** Should only be called in the main Qt thread!
            This slot is automatically called by 'update' (in
    Qt::QueuedConnection mode).
    **/
    void refresh();

signals:

    //! Schedules a call to refresh
    void scheduleRefresh();

protected:
    //! Current progress value (percent)
    QAtomicInt m_currentValue;

    //! Last displayed progress value (percent)
    QAtomicInt m_lastRefreshValue;
};
