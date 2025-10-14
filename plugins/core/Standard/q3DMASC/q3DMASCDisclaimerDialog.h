// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q3DMASC_DISCLAIMER_DIALOG_HEADER
#define Q3DMASC_DISCLAIMER_DIALOG_HEADER

#include <ui_classifyDisclaimerDlg.h>
#include <ui_trainDisclaimerDlg.h>

// qCC_plugins
#include <ecvMainAppInterface.h>

// Qt
#include <QMainWindow>

//! Dialog for displaying the 3DSMAC/UEB disclaimer
class TrainDisclaimerDialog : public QDialog, public Ui::TrainDisclaimerDialog {
public:
    //! Default constructor
    TrainDisclaimerDialog(QWidget* parent = 0)
        : QDialog(parent), Ui::TrainDisclaimerDialog() {
        setupUi(this);

        QString compilationInfo;
        compilationInfo += "Version " + QString(Q3DMASC_VERSION);
        compilationInfo += QStringLiteral("<br><i>Compiled with");

#if defined(_MSC_VER)
        compilationInfo += QStringLiteral(" MSVC %1 and").arg(_MSC_VER);
#endif

        compilationInfo += QStringLiteral(" Qt %1").arg(QT_VERSION_STR);
        compilationInfo += QStringLiteral("</i>");
        compilationInfo += " [cc " + QString(GIT_BRANCH_CC) + "/" +
                           QString(GIT_COMMMIT_HASH_CC) + "]";
        compilationInfo += " [3dmasc " + QString(GIT_TAG_3DMASC) + " " +
                           QString(GIT_BRANCH_3DMASC) + "/" +
                           QString(GIT_COMMMIT_HASH_3DMASC) + "]";

        label_compilationInformation->setText(compilationInfo);
    }
};

// whether disclaimer has already been displayed (and accepted) or not
static bool s_trainDisclaimerAccepted = false;

static bool ShowTrainDisclaimer(ecvMainAppInterface* app) {
    if (!s_trainDisclaimerAccepted) {
        // if the user "cancels" it, then he refuses the diclaimer!
        s_trainDisclaimerAccepted =
                TrainDisclaimerDialog(app ? app->getMainWindow() : 0).exec();
    }

    return s_trainDisclaimerAccepted;
}

//! Dialog for displaying the M3C2/UEB disclaimer
class ClassifyDisclaimerDialog : public QDialog,
                                 public Ui::ClassifyDisclaimerDialog {
public:
    //! Default constructor
    ClassifyDisclaimerDialog(QWidget* parent = 0)
        : QDialog(parent), Ui::ClassifyDisclaimerDialog() {
        setupUi(this);

        QString compilationInfo;
        compilationInfo += "Version " + QString(Q3DMASC_VERSION);
        compilationInfo += QStringLiteral("<br><i>Compiled with");

#if defined(_MSC_VER)
        compilationInfo += QStringLiteral(" MSVC %1 and").arg(_MSC_VER);
#endif

        compilationInfo += QStringLiteral(" Qt %1").arg(QT_VERSION_STR);
        compilationInfo += QStringLiteral("</i>");
        compilationInfo += " [cc " + QString(GIT_BRANCH_CC) + "/" +
                           QString(GIT_COMMMIT_HASH_CC) + "]";
        compilationInfo += " [3dmasc " + QString(GIT_TAG_3DMASC) + " " +
                           QString(GIT_BRANCH_3DMASC) + "/" +
                           QString(GIT_COMMMIT_HASH_3DMASC) + "]";

        label_compilationInformation->setText(compilationInfo);
    }
};

// whether disclaimer has already been displayed (and accepted) or not
static bool s_classifyDisclaimerAccepted = false;

static bool ShowClassifyDisclaimer(ecvMainAppInterface* app) {
    if (!s_classifyDisclaimerAccepted) {
        // if the user "cancels" it, then he refuses the diclaimer!
        s_classifyDisclaimerAccepted =
                ClassifyDisclaimerDialog(app ? app->getMainWindow() : 0).exec();
    }

    return s_classifyDisclaimerAccepted;
}

#endif  // Q3DMASC_DISCLAIMER_DIALOG_HEADER
