// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvCustomViewpointButtonDlg.h"

#include "ui_customViewpointButtonDlg.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVTools.h>

// ECV_DB_TOOL
#include <ecvDisplayTools.h>
#include <ecvGenericCameraTool.h>

#ifdef USE_PCL_BACKEND
#include <../PCLEngine/Tools/EditCameraTool.h>
#endif

// LOCAL
#include <QDebug>
#include <QFileDialog>
#include <QToolButton>
#include <cassert>
#include <sstream>
#include <string>

#include "ecvFileUtils.h"
#include "ecvPersistentSettings.h"
#include "ecvSettingManager.h"

// User interface
//=============================================================================
class pqCustomViewpointButtonDialogUI : public Ui::CustomViewpointButtonDlg {
    struct RowData {
        QPointer<QLabel> IndexLabel;
        QPointer<QLineEdit> ToolTipEdit;
        QPointer<QPushButton> AssignButton;
        QPointer<QToolButton> DeleteButton;
    };

    QPointer<::ecvCustomViewpointButtonDlg> Parent;
    QList<RowData> Rows;

public:
    pqCustomViewpointButtonDialogUI(::ecvCustomViewpointButtonDlg* parent)
        : Parent(parent) {}
    ~pqCustomViewpointButtonDialogUI() { this->setNumberOfRows(0); }

    void setNumberOfRows(int rows) {
        if (this->Rows.size() == rows) {
            return;
        }

        // enable/disable add button.
        this->add->setEnabled(
                rows < ::ecvCustomViewpointButtonDlg::MAXIMUM_NUMBER_OF_ITEMS);

        // remove extra rows.
        for (int cc = this->Rows.size() - 1; cc >= rows; --cc) {
            auto& arow = this->Rows[cc];
            this->gridLayout->removeWidget(arow.IndexLabel);
            this->gridLayout->removeWidget(arow.ToolTipEdit);
            this->gridLayout->removeWidget(arow.AssignButton);
            if (arow.DeleteButton) {
                this->gridLayout->removeWidget(arow.DeleteButton);
            }
            delete arow.IndexLabel;
            delete arow.ToolTipEdit;
            delete arow.AssignButton;
            delete arow.DeleteButton;
            this->Rows.pop_back();
        }

        // add new rows.
        for (int cc = this->Rows.size(); cc < rows; ++cc) {
            RowData arow;
            arow.IndexLabel = new QLabel(QString::number(cc + 1), this->Parent);
            arow.IndexLabel->setAlignment(Qt::AlignCenter);
            arow.ToolTipEdit = new QLineEdit(this->Parent);
            arow.ToolTipEdit->setToolTip(
                    "This text will be set to the buttons tool tip.");
            arow.ToolTipEdit->setText(
                    ::ecvCustomViewpointButtonDlg::DEFAULT_TOOLTIP);
            arow.ToolTipEdit->setObjectName(QString("toolTip%1").arg(cc));
            arow.AssignButton =
                    new QPushButton("Current Viewpoint", this->Parent);
            arow.AssignButton->setProperty(
                    "pqCustomViewpointButtonDialog_INDEX", cc);
            arow.AssignButton->setObjectName(
                    QString("currentViewpoint%1").arg(cc));
            this->Parent->connect(arow.AssignButton, SIGNAL(clicked()),
                                  SLOT(assignCurrentViewpoint()));
            this->gridLayout->addWidget(arow.IndexLabel, cc + 1, 0);
            this->gridLayout->addWidget(arow.ToolTipEdit, cc + 1, 1);
            this->gridLayout->addWidget(arow.AssignButton, cc + 1, 2);
            if (cc >= ::ecvCustomViewpointButtonDlg::MINIMUM_NUMBER_OF_ITEMS) {
                arow.DeleteButton = new QToolButton(this->Parent);
                arow.DeleteButton->setObjectName(QString("delete%1").arg(cc));
                arow.DeleteButton->setIcon(
                        QIcon(":/Resources/images/ecvdelete.png"));
                arow.DeleteButton->setProperty(
                        "pqCustomViewpointButtonDialog_INDEX", cc);
                this->gridLayout->addWidget(arow.DeleteButton, cc + 1, 3);
                this->Parent->connect(arow.DeleteButton, SIGNAL(clicked()),
                                      SLOT(deleteRow()));
            }
            this->Rows.push_back(arow);
        }
    }

    int rowCount() const { return this->Rows.size(); }

    void setToolTips(const QStringList& txts) {
        assert(this->Rows.size() == txts.size());
        for (int cc = 0, max = this->Rows.size(); cc < max; ++cc) {
            this->Rows[cc].ToolTipEdit->setText(txts[cc]);
        }
    }

    QStringList toolTips() const {
        QStringList tips;
        for (const auto& arow : this->Rows) {
            tips.push_back(arow.ToolTipEdit->text());
        }
        return tips;
    }

    void setToolTip(int index, const QString& txt) {
        assert(index >= 0 && index < this->Rows.size());
        this->Rows[index].ToolTipEdit->setText(txt);
        this->Rows[index].ToolTipEdit->selectAll();
        this->Rows[index].ToolTipEdit->setFocus();
    }

    QString toolTip(int index) const {
        assert(index >= 0 && index < this->Rows.size());
        return this->Rows[index].ToolTipEdit->text();
    }

    void deleteRow(int index) {
        assert(index >= 0 && index < this->Rows.size());
        auto& arow = this->Rows[index];
        this->gridLayout->removeWidget(arow.IndexLabel);
        this->gridLayout->removeWidget(arow.ToolTipEdit);
        this->gridLayout->removeWidget(arow.AssignButton);
        if (arow.DeleteButton) {
            this->gridLayout->removeWidget(arow.DeleteButton);
        }
        delete arow.IndexLabel;
        delete arow.ToolTipEdit;
        delete arow.AssignButton;
        delete arow.DeleteButton;
        this->Rows.removeAt(index);

        // now update names and widget layout in the grid
        for (int cc = index; cc < this->Rows.size(); ++cc) {
            auto& currentRow = this->Rows[cc];
            currentRow.IndexLabel->setText(QString::number(cc + 1));
            currentRow.AssignButton->setProperty(
                    "pqCustomViewpointButtonDialog_INDEX", cc);
            this->gridLayout->addWidget(currentRow.IndexLabel, cc + 1, 0);
            this->gridLayout->addWidget(currentRow.ToolTipEdit, cc + 1, 1);
            this->gridLayout->addWidget(currentRow.AssignButton, cc + 1, 2);
            if (currentRow.DeleteButton) {
                currentRow.DeleteButton->setProperty(
                        "pqCustomViewpointButtonDialog_INDEX", cc);
                this->gridLayout->addWidget(currentRow.DeleteButton, cc + 1, 3);
            }
        }

        // enable/disable add button.
        this->add->setEnabled(
                this->Rows.size() <
                ::ecvCustomViewpointButtonDlg::MAXIMUM_NUMBER_OF_ITEMS);
    }
};

// Organizes button config file info in a single location.
//=============================================================================
class pqCustomViewpointButtonFileInfo {
public:
    pqCustomViewpointButtonFileInfo()
        : FileIdentifier("CustomViewpointsConfiguration"),
          FileDescription("Custom Viewpoints Configuration"),
          FileExtension(".cam"),
          WriterVersion("1.0") {}
    const char* FileIdentifier;
    const char* FileDescription;
    const char* FileExtension;
    const char* WriterVersion;
};

//------------------------------------------------------------------------------
const QString ecvCustomViewpointButtonDlg::DEFAULT_TOOLTIP =
        QString("Unnamed Viewpoint");
const int ecvCustomViewpointButtonDlg::MINIMUM_NUMBER_OF_ITEMS = 0;
const int ecvCustomViewpointButtonDlg::MAXIMUM_NUMBER_OF_ITEMS = 30;

//------------------------------------------------------------------------------
ecvCustomViewpointButtonDlg::ecvCustomViewpointButtonDlg(QWidget* Parent,
                                                         Qt::WindowFlags flags,
                                                         QStringList& toolTips,
                                                         QStringList& configs,
                                                         QString& curConfig)
    : QDialog(Parent, flags), ui(nullptr) {
    this->ui = new pqCustomViewpointButtonDialogUI(this);
    this->ui->setupUi(this);
    this->setToolTipsAndConfigurations(toolTips, configs);
    this->setCurrentConfiguration(curConfig);
    QObject::connect(this->ui->add, SIGNAL(clicked()), this, SLOT(appendRow()));
    QObject::connect(this->ui->clearAll, SIGNAL(clicked()), this,
                     SLOT(clearAll()));
    QObject::connect(this->ui->importAll, SIGNAL(clicked()), this,
                     SLOT(importConfigurations()));
    QObject::connect(this->ui->exportAll, SIGNAL(clicked()), this,
                     SLOT(exportConfigurations()));
}

//------------------------------------------------------------------------------
ecvCustomViewpointButtonDlg::~ecvCustomViewpointButtonDlg() {
    delete this->ui;
    this->ui = NULL;
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::setToolTipsAndConfigurations(
        const QStringList& toolTips, const QStringList& configs) {
    if (toolTips.size() != configs.size()) {
        qWarning(
                "`setToolTipsAndConfigurations` called with mismatched "
                "lengths.");
    }

    int minSize = std::min(toolTips.size(), configs.size());
    if (minSize > MAXIMUM_NUMBER_OF_ITEMS) {
        qWarning() << "configs greater than " << MAXIMUM_NUMBER_OF_ITEMS
                   << " will be ignored.";
        minSize = MAXIMUM_NUMBER_OF_ITEMS;
    }

    QStringList realToolTips = toolTips.mid(0, minSize);
    QStringList realConfigs = configs.mid(0, minSize);

    // ensure there are at least MINIMUM_NUMBER_OF_ITEMS items.
    for (int cc = minSize; cc < MINIMUM_NUMBER_OF_ITEMS; ++cc) {
        realToolTips.push_back(DEFAULT_TOOLTIP);
        realConfigs.push_back(QString());
    }

    this->ui->setNumberOfRows(realToolTips.size());
    this->ui->setToolTips(realToolTips);
    this->setConfigurations(realConfigs);
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::setToolTips(const QStringList& toolTips) {
    if (toolTips.length() != this->ui->rowCount()) {
        CVLog::Error("Error: Wrong number of tool tips.");
        return;
    }
    this->ui->setToolTips(toolTips);
}

//------------------------------------------------------------------------------
QStringList ecvCustomViewpointButtonDlg::getToolTips() {
    return this->ui->toolTips();
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::setConfigurations(
        const QStringList& configs) {
    if (configs.length() != this->ui->rowCount()) {
        CVLog::Error("Error: Wrong number of configurations.");
        return;
    }
    this->Configurations = configs;
}

//------------------------------------------------------------------------------
QStringList ecvCustomViewpointButtonDlg::getConfigurations() {
    return this->Configurations;
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::setCurrentConfiguration(
        const QString& config) {
    this->CurrentConfiguration = config;
}

//------------------------------------------------------------------------------
QString ecvCustomViewpointButtonDlg::getCurrentConfiguration() {
    return this->CurrentConfiguration;
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::importConfigurations() {
    // What follows is a reader for an xml format that contains
    // a group of nested Camera Configuration XML hierarchies
    // each written by the vtkSMCameraConfigurationWriter.
    // The nested configuration hierarchies might be empty.
    pqCustomViewpointButtonFileInfo fileInfo;

    QString filters = QString("%1 (*%2);;All Files (*.*)")
                              .arg(fileInfo.FileDescription)
                              .arg(fileInfo.FileExtension);

    // persistent settings
    QString currentPath =
            ecvSettingManager::getValue(ecvPS::LoadFile(), ecvPS::CurrentPath(),
                                        ecvFileUtils::defaultDocPath())
                    .toString();
    QStringList selectedFiles = QFileDialog::getOpenFileNames(
            this, QString("Load Custom Viewpoints Configuration"), currentPath,
            filters);

    if (selectedFiles.isEmpty()) return;
    QString filename;
    filename = selectedFiles[0];

    ecvDisplayTools::LoadCameraParameters(CVTools::FromQString(filename));

#ifdef USE_PCL_BACKEND
    EditCameraTool::UpdateCameraInfo();
#else
    CVLog::Warning(
            "[ecvCustomViewpointButtonDlg::importConfigurations] please use "
            "pcl as backend and then try again!");
#endif

    // read buttons, their toolTips, and configurations.
    QStringList toolTips;
    toolTips << DEFAULT_TOOLTIP;
    QStringList configs;
    configs << ecvGenericCameraTool::CurrentCameraParam.toString().c_str();

    // Pass the newly loaded configuration to the GUI.
    this->setToolTipsAndConfigurations(toolTips, configs);
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::exportConfigurations() {
    pqCustomViewpointButtonFileInfo fileInfo;

    QString filters = QString("%1 (*%2);;All Files (*.*)")
                              .arg(fileInfo.FileDescription)
                              .arg(fileInfo.FileExtension);

    // default output path (+ filename)
    QString currentPath =
            ecvSettingManager::getValue(ecvPS::SaveFile(), ecvPS::CurrentPath(),
                                        ecvFileUtils::defaultDocPath())
                    .toString();

    // ask the user for the output filename
    QString selectedFilename = QFileDialog::getSaveFileName(
            this, tr("Save Custom Viewpoints Configuration"), currentPath,
            filters);

    if (selectedFilename.isEmpty()) {
        // process cancelled by the user
        return;
    }

    QString filename = selectedFilename;
    ecvDisplayTools::SaveCameraParameters(CVTools::FromQString(filename));
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::appendRow() {
    const int numRows = this->ui->rowCount();
    assert(numRows < MAXIMUM_NUMBER_OF_ITEMS);
    this->ui->setNumberOfRows(numRows + 1);
#ifdef USE_PCL_BACKEND
    EditCameraTool::UpdateCameraInfo();
#else
    CVLog::Warning(
            "[ecvCustomViewpointButtonDlg::importConfigurations] please use "
            "pcl as backend and then try again!");
#endif

    // read configurations.
    QString curCameraParam =
            ecvGenericCameraTool::CurrentCameraParam.toString().c_str();
    this->Configurations.push_back(curCameraParam);
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::clearAll() {
    this->setToolTipsAndConfigurations(QStringList(), QStringList());
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::assignCurrentViewpoint() {
    int row = -1;
    if (QObject* asender = this->sender()) {
        row = asender->property("pqCustomViewpointButtonDialog_INDEX").toInt();
    }

    if (row >= 0 && row < this->ui->rowCount()) {
        this->Configurations[row] = this->CurrentConfiguration;
        if (this->ui->toolTip(row) ==
            ecvCustomViewpointButtonDlg::DEFAULT_TOOLTIP) {
            this->ui->setToolTip(
                    row, "Current Viewpoint " + QString::number(row + 1));
        }
    }
}

//------------------------------------------------------------------------------
void ecvCustomViewpointButtonDlg::deleteRow() {
    int row = -1;
    if (QObject* asender = this->sender()) {
        row = asender->property("pqCustomViewpointButtonDialog_INDEX").toInt();
    }

    if (row >= 0 && row < this->ui->rowCount()) {
        this->ui->deleteRow(row);
        this->Configurations.removeAt(row);
    }
}
