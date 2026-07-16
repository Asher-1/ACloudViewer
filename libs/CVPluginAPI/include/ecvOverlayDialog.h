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

class ecvGenericGLDisplay;

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

    //! Bind this dialog to a specific view.
    //! When bound, the dialog operates on the given view regardless of which
    //! view is UI-active.  Pass nullptr to unbind (returns to active-follows).
    void bindToView(ecvGenericGLDisplay* view);

    //! Returns the explicitly bound view, or nullptr (active-follows mode).
    ecvGenericGLDisplay* getBoundView() const { return m_boundView; }

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

    //! Returns the associated window widget (may be null)
    QWidget* getAssociatedWindow() const { return m_associatedWin; }

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

    //! Called when a view is unregistered — stops the dialog if the
    //! unregistered view is our m_boundView (prevents use-after-free).
    void onBoundViewUnregistered(ecvGenericGLDisplay* view);

protected:
    // inherited from QObject
    bool eventFilter(QObject* obj, QEvent* e) override;

    //! Associated (MDI) window
    QWidget* m_associatedWin;

    //! Explicitly bound view (Phase D).  nullptr = follow active view.
    ecvGenericGLDisplay* m_boundView = nullptr;

    //! Running/processing state
    bool m_processing;

    //! Overridden keys
    QList<int> m_overriddenKeys;
};
