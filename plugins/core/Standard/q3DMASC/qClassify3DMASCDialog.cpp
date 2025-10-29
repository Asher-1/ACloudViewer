// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qClassify3DMASCDialog.h"

// qCC_plugins
#include <ecvMainAppInterface.h>

// qCC_db
#include <ecvPointCloud.h>

// Qt
#include <QComboBox>
#include <QMainWindow>
#include <QPushButton>
#include <QSettings>

static ccPointCloud* GetCloudFromCombo(QComboBox* comboBox, ccHObject* dbRoot) {
    assert(comboBox && dbRoot);
    if (!comboBox || !dbRoot) {
        assert(false);
        return 0;
    }

    // return the cloud currently selected in the combox box
    int index = comboBox->currentIndex();
    if (index < 0) {
        assert(false);
        return 0;
    }
    unsigned uniqueID = comboBox->itemData(index).toUInt();
    ccHObject* item = dbRoot->find(uniqueID);
    if (!item || !item->isA(CV_TYPES::POINT_CLOUD)) {
        assert(false);
        return 0;
    }
    return static_cast<ccPointCloud*>(item);
}

Classify3DMASCDialog::Classify3DMASCDialog(ecvMainAppInterface* app,
                                           bool trainMode /*=false*/)
    : QDialog(app ? app->getMainWindow() : 0),
      Ui::Classify3DMASCDialog(),
      m_app(app) {
    setupUi(this);

    if (m_app) {
        // add list of clouds to the combo-boxes
        ccHObject::Container clouds;
        if (m_app->dbRootObject()) {
            m_app->dbRootObject()->filterChildren(clouds, true,
                                                  CV_TYPES::POINT_CLOUD, true);
        }

        unsigned cloudCount = 0;
        for (size_t i = 0; i < clouds.size(); ++i) {
            QString name = clouds[i]->getName() +
                           QString(" [%1]").arg(clouds[i]->getUniqueID());
            QVariant uniqueID(clouds[i]->getUniqueID());
            cloud1ComboBox->addItem(name, uniqueID);
            cloud2ComboBox->addItem(name, uniqueID);
            cloud3ComboBox->addItem(name, uniqueID);
            cloud4ComboBox->addItem(name, uniqueID);
            testCloudComboBox->addItem(name, uniqueID);
            ++cloudCount;
        }

        testCloudComboBox->addItem("", 0);

        // if 3 clouds are loaded, then there's chances that the first one is
        // the global cloud!
        cloud1ComboBox->setCurrentIndex(
                /*cloudCount > 0 ? (cloudCount > 2 ? 1 : 0) : */ -1);
        connect(cloud1ComboBox, SIGNAL(currentIndexChanged(int)), this,
                SLOT(onCloudChanged(int)));
        cloud2ComboBox->setCurrentIndex(
                /*cloudCount > 1 ? (cloudCount > 2 ? 2 : 1) : */ -1);
        connect(cloud2ComboBox, SIGNAL(currentIndexChanged(int)), this,
                SLOT(onCloudChanged(int)));
        cloud3ComboBox->setCurrentIndex(/*cloudCount > 2 ? 0 : */ -1);
        connect(cloud3ComboBox, SIGNAL(currentIndexChanged(int)), this,
                SLOT(onCloudChanged(int)));
        cloud4ComboBox->setCurrentIndex(/*cloudCount > 3 ? 0 : */ -1);
        connect(cloud4ComboBox, SIGNAL(currentIndexChanged(int)), this,
                SLOT(onCloudChanged(int)));
        testCloudComboBox->setCurrentIndex(-1);

        if (cloudCount == 0 && app) {
            app->dispToConsole(QString("You need at least 1 loaded cloud to ") +
                                       (trainMode ? "train a classifier"
                                                  : "classify it..."),
                               ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        }
    }

    if (trainMode) {
        label->setText(tr("Trainer file"));
        warningLabel->setVisible(false);
        warningLabel->setText(
                "Assign each role to the right cloud, and select the cloud on "
                "which to train the classifier");
    }

    onCloudChanged(0);

    readSettings();
}

Classify3DMASCDialog::~Classify3DMASCDialog() { writeSettings(); }

void Classify3DMASCDialog::readSettings() {
    QSettings settings;
    settings.beginGroup("3DMASC");
    bool keepAttributes = settings.value("keepAttributes", false).toBool();
    this->keepAttributesCheckBox->setChecked(keepAttributes);
}

void Classify3DMASCDialog::writeSettings() {
    QSettings settings;
    settings.beginGroup("3DMASC");
    settings.setValue("keepAttributes", keepAttributesCheckBox->isChecked());
}

void Classify3DMASCDialog::setComboBoxIndex(
        const QMap<QString, QString>& rolesAndNames,
        QLabel* label,
        const QMap<QString, QVariant>& namesAndUniqueIds,
        QComboBox* comboBox) {
    QString name = label == testLabel
                           ? QString("TEST")
                           : label->text();  // testLabel is specific, the role
                                             // is always TEST

    QMap<QString, QVariant>::const_iterator it(namesAndUniqueIds.find(name));
    if (it != namesAndUniqueIds.end()) {
        int index = comboBox->findData(it.value());
        if (index != -1) {
            comboBox->setCurrentIndex(index);
        }
    }
}

void Classify3DMASCDialog::setCloudRoles(
        const QList<QString>& roles,
        QString& corePointsLabel,
        const QMap<QString, QString>& rolesAndNames) {
    int index = 0;
    for (const QString& role : roles) {
        if (role != "TEST") {
            switch (index) {
                case 0:
                    cloud1Label->setText(role);
                    if (corePointsLabel
                                .isEmpty())  // if "core_points:" is not in the
                                             // parameter file, the
                                             // corePointsLabel is the first
                                             // encountered role
                    {
                        corePointsLabel = role;
                    }
                    break;
                case 1:
                    cloud2Label->setText(role);
                    break;
                case 2:
                    cloud3Label->setText(role);
                    break;
                case 3:
                    cloud4Label->setText(role);
                    break;
                default:
                    // this dialog can't handle more than 3 roles!
                    break;
            }
            ++index;
        }
    }

    if (index < 1) {
        cloud1ComboBox->setEnabled(false);
    }
    if (index < 2) {
        cloud2ComboBox->setEnabled(false);
        cloud2ComboBox->setVisible(false);
        cloud2Label->setVisible(false);
    }
    if (index < 3) {
        cloud3ComboBox->setEnabled(false);
        cloud3ComboBox->setVisible(false);
        cloud3Label->setVisible(false);
    }
    if (index < 4) {
        cloud4ComboBox->setEnabled(false);
        cloud4ComboBox->setVisible(false);
        cloud4Label->setVisible(false);
    }

    // now we will try to preset the combo boxes depending on the names which
    // are in the parameter file
    QMap<QString, QVariant> namesAndUniqueIds;

    // build a map 'name : uniqueId' of the available clouds in the database
    // tree (duplicate names are not handled, simply keep the first occurrence)
    ccHObject::Container clouds;
    if (m_app->dbRootObject()) {
        m_app->dbRootObject()->filterChildren(clouds, true,
                                              CV_TYPES::POINT_CLOUD);
    }
    for (size_t i = 0; i < clouds.size(); ++i) {
        if (clouds[i]->isA(CV_TYPES::POINT_CLOUD))  // as filterChildren only
                                                    // tests 'isKindOf'
        {
            QVariant uniqueID(clouds[i]->getUniqueID());
            namesAndUniqueIds[clouds[i]->getName().toUpper()] = uniqueID;
        }
    }

    // preset the combo boxes if possible
    setComboBoxIndex(rolesAndNames, cloud1Label, namesAndUniqueIds,
                     cloud1ComboBox);
    setComboBoxIndex(rolesAndNames, cloud2Label, namesAndUniqueIds,
                     cloud2ComboBox);
    setComboBoxIndex(rolesAndNames, cloud3Label, namesAndUniqueIds,
                     cloud3ComboBox);
    setComboBoxIndex(rolesAndNames, cloud4Label, namesAndUniqueIds,
                     cloud4ComboBox);
    setComboBoxIndex(rolesAndNames, testLabel, namesAndUniqueIds,
                     testCloudComboBox);
}

void Classify3DMASCDialog::getClouds(
        QMap<QString, ccPointCloud*>& clouds) const {
    if (!m_app) {
        assert(false);
        return;
    }

    if (cloud1ComboBox->isEnabled()) {
        clouds.insert(cloud1Label->text(),
                      GetCloudFromCombo(cloud1ComboBox, m_app->dbRootObject()));
    }

    if (cloud2ComboBox->isEnabled()) {
        clouds.insert(cloud2Label->text(),
                      GetCloudFromCombo(cloud2ComboBox, m_app->dbRootObject()));
    }

    if (cloud3ComboBox->isEnabled()) {
        clouds.insert(cloud3Label->text(),
                      GetCloudFromCombo(cloud3ComboBox, m_app->dbRootObject()));
    }

    if (cloud4ComboBox->isEnabled()) {
        clouds.insert(cloud4Label->text(),
                      GetCloudFromCombo(cloud4ComboBox, m_app->dbRootObject()));
    }

    if (testCloudComboBox->currentIndex() >= 0) {
        clouds.insert("TEST", GetCloudFromCombo(testCloudComboBox,
                                                m_app->dbRootObject()));
    }
}

void Classify3DMASCDialog::onCloudChanged(int dummy) {
    if (!cloud1ComboBox->isEnabled()) {
        // this means that no role has been defined yet
        buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
        return;
    }

    buttonBox->button(QDialogButtonBox::Ok)
            ->setEnabled(cloud1ComboBox->currentIndex() >= 0);
}
