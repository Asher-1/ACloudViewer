// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CVPluginAPI.h"

// Qt
#include <QDialog>
#include <QList>

//! Generic overlay dialog interface
class CVPLUGIN_LIB_API ccOverlayDialog : public QDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccOverlayDialog(QWidget* parent = nullptr,
                             Qt::WindowFlags flags = Qt::FramelessWindowHint |
                                                     Qt::Tool);

    //! Destructor
    ~ccOverlayDialog() override;

    //! Links the overlay dialog with a MDI window
    /** Warning: link can't be modified while dialog is displayed/process is
    running! \return success
    **/
    virtual bool linkWith(QWidget* win);

    //! Starts process
    /** \return success
     **/
    virtual bool start();

    //! Stops process/dialog
    /** Automatically emits the 'processFinished' signal (with input state as
    argument). \param accepted process/dialog result
    **/
    virtual void stop(bool accepted);

    // reimplemented from QDialog
    void reject() override;

    //! Adds a keyboard shortcut (single key) that will be overridden from the
    //! associated window
    /** When an overridden key is pressed, the shortcutTriggered(int) signal is
     *emitted.
     **/
    void addOverridenShortcut(Qt::Key key);

    //! Returns whether the tool is currently started or not
    bool started() const { return m_processing; }

signals:

    //! Signal emitted when process is finished
    /** \param accepted specifies how the process finished (accepted or not)
     **/
    void processFinished(bool accepted);

    //! Signal emitted when an overridden key shortcut is pressed
    /** See ccOverlayDialog::addOverridenShortcut
     **/
    void shortcutTriggered(int key);

    //! Signal emitted when a 'show' event is detected
    void shown();

protected slots:

    //! Slot called when the linked window is deleted (calls 'onClose')
    virtual void onLinkedWindowDeletion(QObject* object = nullptr);

protected:
    // inherited from QObject
    bool eventFilter(QObject* obj, QEvent* e) override;

    //! Associated (MDI) window
    QWidget* m_associatedWin;

    //! Running/processing state
    bool m_processing;

    //! Overridden keys
    QList<int> m_overriddenKeys;
};
