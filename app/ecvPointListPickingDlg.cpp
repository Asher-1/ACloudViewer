// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPointListPickingDlg.h"

// Qt
#include <QApplication>
#include <QClipboard>
#include <QFileDialog>
#include <QMenu>
#include <QMessageBox>
#include <QSettings>

// cloudViewer
#include <CVConst.h>
#include <CVLog.h>

// CV_DB_LIB
#include <ecv2DLabel.h>
#include <ecvGenericGLDisplay.h>
#include <ecvGenericMesh.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

// qCC_io
#include <AsciiFilter.h>

// local
#include "MainWindow.h"
#include "db_tree/ecvDBRoot.h"

// system
#include <assert.h>

#include <algorithm>
#include <cmath>

// semi persistent settings
static unsigned s_pickedPointsStartIndex = 0;
static bool s_showGlobalCoordsCheckBoxChecked = false;
static const char s_pickedPointContainerName[] = "Picked points list";
static const char s_defaultLabelBaseName[] = "Point #";

ccPointListPickingDlg::ccPointListPickingDlg(ccPickingHub* pickingHub,
                                             QWidget* parent)
    : ccPointPickingGenericInterface(pickingHub, parent),
      Ui::PointListPickingDlg(),
      m_associatedEntity(nullptr),
      m_lastPreviousID(0),
      m_orderedLabelsContainer(nullptr) {
    setupUi(this);

    exportToolButton->setPopupMode(QToolButton::MenuButtonPopup);
    QMenu* menu = new QMenu(exportToolButton);
    QAction* exportASCII_xyz = menu->addAction("x,y,z");
    QAction* exportASCII_ixyz = menu->addAction("local index,x,y,z");
    QAction* exportASCII_gxyz = menu->addAction("global index,x,y,z");
    QAction* exportASCII_lxyz = menu->addAction("label name,x,y,z");
    QAction* exportToNewCloud = menu->addAction("new cloud");
    QAction* exportToNewPolyline = menu->addAction("new polyline");
    exportToolButton->setMenu(menu);

    tableWidget->verticalHeader()->setSectionResizeMode(
            QHeaderView::ResizeToContents);

    startIndexSpinBox->setValue(s_pickedPointsStartIndex);
    showGlobalCoordsCheckBox->setChecked(s_showGlobalCoordsCheckBoxChecked);

    connect(cancelToolButton, &QAbstractButton::clicked, this,
            &ccPointListPickingDlg::cancelAndExit);
    connect(revertToolButton, &QAbstractButton::clicked, this,
            &ccPointListPickingDlg::removeLastEntry);
    connect(validToolButton, &QAbstractButton::clicked, this,
            &ccPointListPickingDlg::applyAndExit);
    connect(exportToolButton, &QAbstractButton::clicked, exportToolButton,
            &QToolButton::showMenu);
    connect(exportASCII_xyz, &QAction::triggered, this,
            &ccPointListPickingDlg::exportToASCII_xyz);
    connect(exportASCII_ixyz, &QAction::triggered, this,
            &ccPointListPickingDlg::exportToASCII_ixyz);
    connect(exportASCII_gxyz, &QAction::triggered, this,
            &ccPointListPickingDlg::exportToASCII_gxyz);
    connect(exportASCII_lxyz, &QAction::triggered, this,
            &ccPointListPickingDlg::exportToASCII_lxyz);
    connect(exportToNewCloud, &QAction::triggered, this,
            &ccPointListPickingDlg::exportToNewCloud);
    connect(exportToNewPolyline, &QAction::triggered, this,
            &ccPointListPickingDlg::exportToNewPolyline);

    connect(markerSizeSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccPointListPickingDlg::markerSizeChanged);
    connect(startIndexSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccPointListPickingDlg::startIndexChanged);

    connect(showGlobalCoordsCheckBox, &QAbstractButton::clicked, this,
            &ccPointListPickingDlg::updateList);

    updateList();
}

unsigned ccPointListPickingDlg::getPickedPoints(
        std::vector<cc2DLabel*>& pickedPoints) {
    pickedPoints.clear();

    if (m_orderedLabelsContainer) {
        // get all labels
        ccHObject::Container labels;
        unsigned count = m_orderedLabelsContainer->filterChildren(
                labels, false, CV_TYPES::LABEL_2D);

        try {
            pickedPoints.reserve(count);
        } catch (const std::bad_alloc&) {
            CVLog::Error("Not enough memory!");
            return 0;
        }
        for (unsigned i = 0; i < count; ++i) {
            // Warning: cc2DViewportLabel is also a kind of
            // 'CV_TYPES::LABEL_2D'!
            if (labels[i]->isA(CV_TYPES::LABEL_2D)) {
                cc2DLabel* label = static_cast<cc2DLabel*>(labels[i]);
                if (label->isVisible() && label->size() == 1) {
                    pickedPoints.push_back(label);
                }
            }
        }
    }

    return static_cast<unsigned>(pickedPoints.size());
}

void ccPointListPickingDlg::linkWithEntity(ccHObject* entity) {
    if (!entity && m_associatedEntity) {
        ccDBRoot* dbRoot = MainWindow::TheInstance()
                                   ? MainWindow::TheInstance()->db()
                                   : nullptr;
        if (dbRoot) {
            std::vector<removeInfo> rmInfos;
            if (m_orderedLabelsContainer) {
                if (!m_toBeAdded.empty()) {
                    for (auto* obj : m_toBeAdded) {
                        obj->getTypeID_recursive(rmInfos, true);
                    }
                    dbRoot->removeElements(m_toBeAdded);
                }
                for (size_t j = 0; j < m_toBeDeleted.size(); ++j) {
                    m_toBeDeleted[j]->setRedrawFlagRecursive(true);
                    m_toBeDeleted[j]->setEnabled(true);
                }
                if (m_orderedLabelsContainer->getChildrenNumber() == 0) {
                    m_orderedLabelsContainer->getTypeID_recursive(rmInfos,
                                                                  true);
                    dbRoot->removeElement(m_orderedLabelsContainer);
                }
            }
            if (!rmInfos.empty()) {
                ecvViewManager::instance().setRemoveViewIds(rmInfos);
            }
        }
        m_toBeDeleted.resize(0);
        m_toBeAdded.resize(0);
        m_orderedLabelsContainer = nullptr;
        { ecvRedrawScope scope; }
    }

    m_associatedEntity = entity;
    m_lastPreviousID = 0;

    if (m_associatedEntity) {
        // find default container
        m_orderedLabelsContainer = nullptr;
        ccHObject::Container groups;
        m_associatedEntity->filterChildren(groups, true,
                                           CV_TYPES::HIERARCHY_OBJECT);

        for (ccHObject::Container::const_iterator it = groups.begin();
             it != groups.end(); ++it) {
            if ((*it)->getName() == s_pickedPointContainerName) {
                m_orderedLabelsContainer = *it;
                break;
            }
        }

        std::vector<cc2DLabel*> previousPickedPoints;
        unsigned count = getPickedPoints(previousPickedPoints);
        // find highest unique ID among the VISIBLE labels
        for (unsigned i = 0; i < count; ++i) {
            m_lastPreviousID = std::max(m_lastPreviousID,
                                        previousPickedPoints[i]->getUniqueID());
        }
    }

    ccShiftedObject* shifted = ccHObjectCaster::ToShifted(entity);
    showGlobalCoordsCheckBox->setEnabled(shifted ? shifted->isShifted()
                                                 : false);
    updateList();
}

void ccPointListPickingDlg::stop(bool state) {
    if (m_associatedEntity) {
        linkWithEntity(nullptr);
    }
    ccPointPickingGenericInterface::stop(state);
}

void ccPointListPickingDlg::cancelAndExit() {
    linkWithEntity(nullptr);
    stop(false);
}

void ccPointListPickingDlg::exportToNewCloud() {
    if (!m_associatedEntity) return;

    // get all labels
    std::vector<cc2DLabel*> labels;
    unsigned count = getPickedPoints(labels);
    if (count != 0) {
        ccPointCloud* cloud = new ccPointCloud();
        if (cloud->reserve(count)) {
            cloud->setName("Picking list");
            for (unsigned i = 0; i < count; ++i) {
                const cc2DLabel::PickedPoint& PP = labels[i]->getPickedPoint(0);
                cloud->addPoint(PP.getPointPosition());
            }

            ccShiftedObject* shifted =
                    ccHObjectCaster::ToShifted(m_associatedEntity);
            if (shifted) {
                cloud->copyGlobalShiftAndScale(*shifted);
            }
            MainWindow::TheInstance()->addToDB(cloud);
        } else {
            CVLog::Error(
                    "Can't export picked points as point cloud: not enough "
                    "memory!");
            delete cloud;
            cloud = nullptr;
        }
    } else {
        CVLog::Error("Pick some points first!");
    }
}

void ccPointListPickingDlg::exportToNewPolyline() {
    if (!m_associatedEntity) return;

    // get all labels
    std::vector<cc2DLabel*> labels;
    unsigned count = getPickedPoints(labels);
    if (count > 1) {
        // we create an "independent" polyline
        ccPointCloud* vertices = new ccPointCloud("vertices");
        ccPolyline* polyline = new ccPolyline(vertices);

        if (!vertices->reserve(count) || !polyline->reserve(count)) {
            CVLog::Error("Not enough memory!");
            delete vertices;
            delete polyline;
            return;
        }

        for (unsigned i = 0; i < count; ++i) {
            const cc2DLabel::PickedPoint& PP = labels[i]->getPickedPoint(0);
            vertices->addPoint(PP.getPointPosition());
        }
        polyline->addPointIndex(0, count);
        polyline->setVisible(true);
        vertices->setEnabled(false);
        ccShiftedObject* shifted =
                ccHObjectCaster::ToShifted(m_associatedEntity);
        if (shifted) {
            polyline->copyGlobalShiftAndScale(*shifted);
        }
        polyline->addChild(vertices);
        MainWindow::TheInstance()->addToDB(polyline);
    } else {
        CVLog::Error("Pick at least two points!");
    }
}

void ccPointListPickingDlg::applyAndExit() {
    if (m_associatedEntity && !m_toBeDeleted.empty()) {
        // apply modifications
        // no need to redraw as they should already be invisible
        MainWindow::TheInstance()->db()->removeElements(m_toBeDeleted);
        m_associatedEntity = nullptr;
    }

    m_toBeDeleted.resize(0);
    m_toBeAdded.resize(0);
    m_orderedLabelsContainer = nullptr;

    updateList();

    stop(true);
}

void ccPointListPickingDlg::removeLastEntry() {
    if (!m_associatedEntity) return;

    // get all labels
    std::vector<cc2DLabel*> labels;
    unsigned count = getPickedPoints(labels);
    if (count == 0) return;

    ccHObject* lastVisibleLabel = labels.back();
    if (lastVisibleLabel->getUniqueID() <= m_lastPreviousID) {
        // old label: hide it and add it to the 'to be deleted' list
        // (will be restored if process is cancelled)
        lastVisibleLabel->setEnabled(false);
        m_toBeDeleted.push_back(lastVisibleLabel);
    } else {
        if (!m_toBeAdded.empty()) {
            assert(m_toBeAdded.back() == lastVisibleLabel);
            m_toBeAdded.pop_back();
        }

        if (m_orderedLabelsContainer) {
            if (lastVisibleLabel->getParent()) {
                lastVisibleLabel->getParent()->removeDependencyWith(
                        lastVisibleLabel);
                lastVisibleLabel->removeDependencyWith(
                        lastVisibleLabel->getParent());
            }
            MainWindow::TheInstance()->db()->removeElement(lastVisibleLabel);
        } else {
            m_associatedEntity->detachChild(lastVisibleLabel);
        }
    }

    updateList();

    { ecvRedrawScope scope; }
}

void ccPointListPickingDlg::clearLastLabel(ccHObject* lastVisibleLabel) {
    // remove last visible label from rendering window
    removeEntity(lastVisibleLabel);

    // remove last visible label from db tree
    if (lastVisibleLabel->getParent()) {
        lastVisibleLabel->getParent()->removeDependencyWith(lastVisibleLabel);
        lastVisibleLabel->removeDependencyWith(lastVisibleLabel->getParent());
    }
    MainWindow::TheInstance()->db()->removeElement(lastVisibleLabel);
}

void ccPointListPickingDlg::removeEntity(ccHObject* lastVisibleLabel) {
    cc2DLabel* label = ccHObjectCaster::To2DLabel(lastVisibleLabel);
    if (label) {
        label->setEnabled(false);
        label->updateLabel();
    }
}

void ccPointListPickingDlg::startIndexChanged(int value) {
    unsigned int uValue = static_cast<unsigned int>(value);

    if (uValue != s_pickedPointsStartIndex) {
        s_pickedPointsStartIndex = uValue;

        updateList();

        { ecvRedrawScope scope(true, false); }
    }
}

void ccPointListPickingDlg::markerSizeChanged(int size) {
    if (size < 1) return;

    ecvGenericGLDisplay* view = ecvViewManager::instance().getEffectiveView();
    if (!view) return;

    ecvGui::ParamStruct guiParams = view->getDisplayParameters();

    if (guiParams.labelMarkerSize != static_cast<unsigned>(size)) {
        guiParams.labelMarkerSize = static_cast<unsigned>(size);
        view->setDisplayParameters(guiParams, true);
        { ecvRedrawScope scope; }
    }
}

void ccPointListPickingDlg::exportToASCII(ExportFormat format) {
    if (!m_associatedEntity) return;

    // get all labels
    std::vector<cc2DLabel*> labels;
    unsigned count = getPickedPoints(labels);
    if (count == 0) return;

    QSettings settings;
    settings.beginGroup("PointListPickingDlg");
    QString filename =
            settings.value("filename", "picking_list.txt").toString();
    settings.endGroup();

    filename = QFileDialog::getSaveFileName(this, "Export to ASCII", filename,
                                            AsciiFilter::GetFileFilter());

    if (filename.isEmpty()) return;

    settings.beginGroup("PointListPickingDlg");
    settings.setValue("filename", filename);
    settings.endGroup();

    FILE* fp = fopen(qPrintable(filename), "wt");
    if (!fp) {
        CVLog::Error(
                QString("Failed to open file '%1' for saving!").arg(filename));
        return;
    }

    // if a global shift exists, ask the user if it should be applied
    CCVector3d shift(0, 0, 0);
    double scale = 1.0;
    ccGenericPointCloud* asCloud =
            ccHObjectCaster::ToGenericPointCloud(m_associatedEntity);
    if (asCloud) {
        shift = asCloud->getGlobalShift();
        scale = asCloud->getGlobalScale();
    }

    if (shift.norm2() != 0 || scale != 1.0) {
        if (QMessageBox::warning(this, "Apply global shift",
                                 "Do you want to apply global shift/scale to "
                                 "exported points?",
                                 QMessageBox::Yes | QMessageBox::No,
                                 QMessageBox::Yes) == QMessageBox::No) {
            // reset shift
            shift = CCVector3d(0, 0, 0);
            scale = 1.0;
        }
    }

    // starting index
    unsigned startIndex =
            static_cast<unsigned>(std::max(0, startIndexSpinBox->value()));

    for (unsigned i = 0; i < count; ++i) {
        assert(labels[i]->size() == 1);
        const cc2DLabel::PickedPoint& PP = labels[i]->getPickedPoint(0);
        const CCVector3* P = PP.cloud->getPoint(PP.index);

        switch (format) {
            case PLP_ASCII_EXPORT_IXYZ:
                fprintf(fp, "%u,", i + startIndex);
                break;
            case PLP_ASCII_EXPORT_GXYZ:
                fprintf(fp, "%u,", PP.index);
                break;
            case PLP_ASCII_EXPORT_LXYZ:
                fprintf(fp, "%s,", qPrintable(labels[i]->getName()));
                break;
            default:
                // nothing to do
                break;
        }

        fprintf(fp, "%.12f,%.12f,%.12f\n",
                static_cast<double>(P->x) / scale - shift.x,
                static_cast<double>(P->y) / scale - shift.y,
                static_cast<double>(P->z) / scale - shift.z);
    }

    fclose(fp);

    CVLog::Print(QString("[I/O] File '%1' saved successfully").arg(filename));
}

void ccPointListPickingDlg::updateList() {
    // get all labels
    std::vector<cc2DLabel*> labels;
    unsigned count = getPickedPoints(labels);

    revertToolButton->setEnabled(count);
    validToolButton->setEnabled(count);
    exportToolButton->setEnabled(count);
    countLineEdit->setText(QString::number(count));
    tableWidget->setRowCount(count);

    if (!count) return;

    // starting index
    int startIndex = startIndexSpinBox->value();
    int precision = 6;
    if (ecvViewManager::instance().activeWidget()) {
        if (auto* effView = ecvViewManager::instance().getEffectiveView()) {
            precision = effView->getDisplayParameters().displayedNumPrecision;
        }
    }

    bool showAbsolute = showGlobalCoordsCheckBox->isEnabled() &&
                        showGlobalCoordsCheckBox->isChecked();

    for (unsigned i = 0; i < count; ++i) {
        const cc2DLabel::PickedPoint& PP = labels[i]->getPickedPoint(0);
        CCVector3 P = PP.getPointPosition();
        CCVector3d Pd = (showAbsolute ? PP.cloudOrVertices()->toGlobal3d(P)
                                      : CCVector3d::fromArray(P.u));

        // point index in list
        tableWidget->setVerticalHeaderItem(
                i, new QTableWidgetItem(QString("%1").arg(i + startIndex)));
        // update name as well
        //  DGM: we don't change the name of old labels that have a non-default
        //  name
        if (labels[i]->getUniqueID() > m_lastPreviousID ||
            labels[i]->getName().startsWith(s_defaultLabelBaseName)) {
            labels[i]->setName(s_defaultLabelBaseName +
                               QString::number(i + startIndex));
        }
        // point absolute index (in cloud)
        tableWidget->setItem(i, 0,
                             new QTableWidgetItem(QString("%1").arg(PP.index)));

        for (unsigned j = 0; j < 3; ++j)
            tableWidget->setItem(i, j + 1,
                                 new QTableWidgetItem(QString("%1").arg(
                                         Pd.u[j], 0, 'f', precision)));
    }

    tableWidget->scrollToBottom();
}

void ccPointListPickingDlg::processPickedPoint(const PickedItem& picked) {
    CVLog::PrintDebug("[PointPicking] processPickedPoint() ENTER");
    if (!picked.entity || picked.entity != m_associatedEntity ||
        !MainWindow::TheInstance())
        return;

    cc2DLabel* newLabel = new cc2DLabel();
    bool addOk = false;
    if (picked.entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        addOk = newLabel->addPickedPoint(
                static_cast<ccGenericPointCloud*>(picked.entity),
                picked.itemIndex, picked.entityCenter);
    } else if (picked.entity->isKindOf(CV_TYPES::MESH)) {
        ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(picked.entity);
        if (mesh && picked.itemIndex < mesh->size()) {
            CCVector3 A, B, C;
            mesh->getTriangleVertices(picked.itemIndex, A, B, C);
            CCVector3 v0 = A - C, v1 = B - C, v2 = picked.P3D - C;
            double d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1);
            double d20 = v2.dot(v0), d21 = v2.dot(v1);
            double denom = d00 * d11 - d01 * d01;
            CCVector2d uv(0, 0);
            if (std::abs(denom) > 1.0e-12) {
                uv.x = (d11 * d20 - d01 * d21) / denom;
                uv.y = (d00 * d21 - d01 * d20) / denom;
            }
            addOk = newLabel->addPickedPoint(mesh, picked.itemIndex, uv,
                                             picked.entityCenter);
        }
    }
    if (!addOk) {
        delete newLabel;
        return;
    }
    newLabel->setVisible(true);
    newLabel->setDisplayedIn2D(false);
    newLabel->displayPointLegend(true);
    newLabel->setCollapsed(true);
    {
        ecvGenericGLDisplay* pickView = picked.pickView;
        if (!pickView) pickView = ecvViewManager::instance().getActiveView();
        if (pickView) {
            newLabel->setDisplay(pickView);
        } else if (picked.entity && picked.entity->getDisplay()) {
            newLabel->setDisplay(picked.entity->getDisplay());
        }
    }
    QSize size(1, 1);
    if (ecvGenericGLDisplay* labelDisplay = newLabel->getDisplay()) {
        if (QWidget* labelWidget = labelDisplay->asWidget()) {
            size = labelWidget->size();
        }
    } else if (QWidget* w = ecvViewManager::instance().activeWidget()) {
        size = w->size();
    }
    if (size.width() <= 0 || size.height() <= 0) {
        size = QSize(1, 1);
    }

    newLabel->setPosition(
            static_cast<float>(picked.clickPoint.x() + 20) / size.width(),
            static_cast<float>(picked.clickPoint.y() + 20) / size.height());

    // add default container if necessary
    if (!m_orderedLabelsContainer) {
        m_orderedLabelsContainer = new ccHObject(s_pickedPointContainerName);
        m_associatedEntity->addChild(m_orderedLabelsContainer);
        m_orderedLabelsContainer->setDisplay(newLabel->getDisplay());
        m_orderedLabelsContainer->setEnabled(true);
        m_orderedLabelsContainer->setVisible(true);
        MainWindow::TheInstance()->addToDB(m_orderedLabelsContainer, false,
                                           true, false, false);
    } else if (newLabel->getDisplay() &&
               m_orderedLabelsContainer->getDisplay() !=
                       newLabel->getDisplay()) {
        m_orderedLabelsContainer->setDisplay(newLabel->getDisplay());
    }
    assert(m_orderedLabelsContainer);
    m_orderedLabelsContainer->addChild(newLabel);
    MainWindow::TheInstance()->addToDB(newLabel, false, true, false, false);
    m_toBeAdded.push_back(newLabel);

    // automatically send the new point coordinates to the clipboard
    QClipboard* clipboard = QApplication::clipboard();
    if (clipboard) {
        CCVector3 P = newLabel->getPickedPoint(0).getPointPosition();
        int precision = 6;
        if (ecvViewManager::instance().activeWidget()) {
            if (auto* effView = ecvViewManager::instance().getEffectiveView()) {
                precision =
                        effView->getDisplayParameters().displayedNumPrecision;
            }
        }
        int indexInList =
                startIndexSpinBox->value() +
                static_cast<int>(
                        m_orderedLabelsContainer->getChildrenNumber()) -
                1;
        clipboard->setText(QString("CC_POINT_#%0(%1;%2;%3)")
                                   .arg(indexInList)
                                   .arg(P.x, 0, 'f', precision)
                                   .arg(P.y, 0, 'f', precision)
                                   .arg(P.z, 0, 'f', precision));
    }

    updateList();

    if (newLabel) {
        newLabel->setRedraw(true);
        newLabel->updateLabel();
        { ecvRedrawScope scope(false, true); }
    }
}
