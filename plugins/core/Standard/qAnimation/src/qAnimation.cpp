//##########################################################################
//#                                                                        #
//#                   CLOUDCOMPARE PLUGIN: qAnimation                      #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#             COPYRIGHT: Ryan Wicks, 2G Robotics Inc., 2015              #
//#                                                                        #
//##########################################################################

#include "qAnimation.h"

// Local
#include "qAnimationDlg.h"

// CORE_LIB
#include <CVTools.h>

// ECV_DB_LIB
#include <ecv2DViewportObject.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvDisplayTools.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// Qt
#include <QMainWindow>
#include <QtGui>

typedef std::vector<cc2DViewportObject *> ViewPortList;
typedef std::vector<ccMesh *> MeshList;

static void GetSelectedObjects(const ccHObject::Container &selectedEntities,
                               CV_CLASS_ENUM type,
                               ccHObject::Container &targetObj) {
    targetObj.clear();
    for (ccHObject *object : selectedEntities) {
        if (object->isKindOf(type)) {
            targetObj.push_back(object);
            continue;
        }

        ccHObject::Container internalContainer;
        object->filterChildren(internalContainer, true, type);
        if (!internalContainer.empty()) {
            targetObj.insert(targetObj.end(), internalContainer.begin(),
                             internalContainer.end());
        }
    }
}

static ViewPortList GetSelectedViewPorts(
        const ccHObject::Container &selectedEntities) {
    ccHObject::Container targetObjContainer;
    GetSelectedObjects(selectedEntities, CV_TYPES::VIEWPORT_2D_OBJECT,
                       targetObjContainer);

    ViewPortList viewports;
    for (auto obj : targetObjContainer) {
        auto *viewport = dynamic_cast<cc2DViewportObject *>(obj);
        if (viewport) {
            viewports.push_back(viewport);
        }
    }
    return viewports;
}

static MeshList GetSelectedMeshes(
        const ccHObject::Container &selectedEntities) {
    ccHObject::Container targetObjContainer;
    GetSelectedObjects(selectedEntities, CV_TYPES::MESH, targetObjContainer);

    MeshList meshes;
    for (auto obj : targetObjContainer) {
        auto *mesh = dynamic_cast<ccMesh *>(obj);
        if (mesh && mesh->hasTextures()) {
            meshes.push_back(mesh);
        }
    }
    return meshes;
}

qAnimation::qAnimation(QObject *parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qAnimation/info.json"),
      m_action(nullptr) {}

void qAnimation::onNewSelection(const ccHObject::Container &selectedEntities) {
    if (m_action == nullptr) {
        return;
    }

    ViewPortList viewports = GetSelectedViewPorts(selectedEntities);
    MeshList meshes = GetSelectedMeshes(m_app->getSelectedEntities());

    if (viewports.size() >= 2 || !meshes.empty()) {
        m_action->setEnabled(true);
        m_action->setToolTip(getDescription());
    } else {
        m_action->setEnabled(false);
        m_action->setToolTip(tr("%1\nAt least 2 viewports must be selected.")
                                     .arg(getDescription()));
    }
}

QList<QAction *> qAnimation::getActions() {
    // default action (if it has not been already created, it's the moment to do
    // it)
    if (!m_action) {
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());

        connect(m_action, &QAction::triggered, this, &qAnimation::doAction);
    }

    return QList<QAction *>{m_action};
}

// what to do when clicked.
void qAnimation::doAction() {
    // m_app should have already been initialized by CC when plugin is loaded!
    //(--> pure internal check)
    assert(m_app);
    if (!m_app) return;

    // get active GL window
    QWidget *glWindow = m_app->getActiveWindow();
    if (!glWindow) {
        m_app->dispToConsole("No active 3D view!",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    ViewPortList viewports = GetSelectedViewPorts(m_app->getSelectedEntities());

    MeshList meshes = GetSelectedMeshes(m_app->getSelectedEntities());

    // store history texture files
    std::vector<std::vector<std::string>> textureFileList;
    for (auto &mesh : meshes) {
        auto *materials = const_cast<ccMaterialSet *>(mesh->getMaterialSet());
        std::vector<std::string> textureFiles;
        if (materials) {
            for (int i = 0; i < materials->size(); ++i) {
                textureFiles.push_back(CVTools::FromQString(
                        materials->at(i)->getTextureFilename()));
            }
        }
        textureFileList.push_back(textureFiles);
    }

    Q_ASSERT((viewports.size() >= 2 ||
              !meshes.empty()));  // action will not be active unless we
                                  // have at least 2 viewports or have meshes

    m_app->dispToConsole(QString("[qAnimation] Selected viewports: %1")
                                 .arg(viewports.size()));
    m_app->dispToConsole(
            QString("[qAnimation] Selected meshes: %1").arg(meshes.size()));

    qAnimationDlg videoDlg(glWindow, m_app->getMainWindow());

    if (!videoDlg.init(viewports, meshes)) {
        m_app->dispToConsole(
                "Failed to initialize the plugin dialog (not enough memory?)",
                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    videoDlg.exec();

    // restore
    for (std::size_t i = 0; i < meshes.size(); ++i) {
        auto &mesh = meshes[i];
        if (mesh && mesh->getMaterialSet()) {
            if (!mesh->updateTextures(textureFileList[i])) {
                CVLog::Warning(QString("Restore texture for %1 failed!")
                                       .arg(mesh->getName()));
            };
        }
    }
    ecvDisplayTools::UpdateScreen();

    // Export trajectory (for debug)
    if (videoDlg.exportTrajectoryOnExit() && videoDlg.getTrajectory()) {
        ccPolyline *trajectory = new ccPolyline(*videoDlg.getTrajectory());
        if (!trajectory) {
            CVLog::Error("Not enough memory");
        } else {
            trajectory->setColor(ecvColor::yellow);
            trajectory->showColors(true);
            trajectory->setWidth(2);

            getMainAppInterface()->addToDB(trajectory);
        }
    }
}
