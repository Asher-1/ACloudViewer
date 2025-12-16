// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvColorScaleEditorDlg.h"

#include "ui_colorScaleEditorDlg.h"

// local
#include "ecvColorScaleEditorWidget.h"
#include "ecvPersistentSettings.h"

// common
#include <ecvMainAppInterface.h>
#include <ecvQtHelpers.h>

// ECV_DB_LIB
#include <ecvColorScalesManager.h>
#include <ecvFileUtils.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

// Qt
#include <QColorDialog>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QInputDialog>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

#include <QMessageBox>
#include <QPlainTextEdit>
#include <QSettings>
#include <QUuid>

// System
#include <cassert>

static char s_defaultEmptyCustomListText[] = "(auto)";

ccColorScaleEditorDialog::ccColorScaleEditorDialog(
        ccColorScalesManager* manager,
        ecvMainAppInterface* mainApp,
        ccColorScale::Shared currentScale /*=0*/,
        QWidget* parent /*=0*/)
    : QDialog(parent),
      m_manager(manager),
      m_colorScale(currentScale),
      m_scaleWidget(new ccColorScaleEditorWidget(this, Qt::Horizontal)),
      m_associatedSF(nullptr),
      m_modified(false),
      m_minAbsoluteVal(0.0),
      m_maxAbsoluteVal(1.0),
      m_mainApp(mainApp),
      m_ui(new Ui::ColorScaleEditorDlg) {
    assert(m_manager);

    m_ui->setupUi(this);

    m_ui->colorScaleEditorFrame->setLayout(new QHBoxLayout());
    m_ui->colorScaleEditorFrame->layout()->setContentsMargins(0, 0, 0, 0);
    m_ui->colorScaleEditorFrame->layout()->addWidget(m_scaleWidget);

    // main combo box
    connect(m_ui->rampComboBox, SIGNAL(activated(int)), this,
            SLOT(colorScaleChanged(int)));

    // import/export buttons
    connect(m_ui->exportToolButton, SIGNAL(clicked()), this,
            SLOT(exportCurrentScale()));
    connect(m_ui->importToolButton, SIGNAL(clicked()), this,
            SLOT(importScale()));

    // upper buttons
    connect(m_ui->renameToolButton, SIGNAL(clicked()), this,
            SLOT(renameCurrentScale()));
    connect(m_ui->saveToolButton, SIGNAL(clicked()), this,
            SLOT(saveCurrentScale()));
    connect(m_ui->deleteToolButton, SIGNAL(clicked()), this,
            SLOT(deleteCurrentScale()));
    connect(m_ui->copyToolButton, SIGNAL(clicked()), this,
            SLOT(copyCurrentScale()));
    connect(m_ui->newToolButton, SIGNAL(clicked()), this,
            SLOT(createNewScale()));
    connect(m_ui->scaleModeComboBox, SIGNAL(activated(int)), this,
            SLOT(relativeModeChanged(int)));

    // scale widget
    connect(m_scaleWidget, SIGNAL(stepSelected(int)), this,
            SLOT(onStepSelected(int)));
    connect(m_scaleWidget, SIGNAL(stepModified(int)), this,
            SLOT(onStepModified(int)));

    // slider editor
    connect(m_ui->deleteSliderToolButton, SIGNAL(clicked()), this,
            SLOT(deletecSelectedStep()));
    connect(m_ui->colorToolButton, SIGNAL(clicked()), this,
            SLOT(changeSelectedStepColor()));
    connect(m_ui->valueDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(changeSelectedStepValue(double)));

    // labels list widget
    connect(m_ui->customLabelsGroupBox, SIGNAL(toggled(bool)), this,
            SLOT(toggleCustomLabelsList(bool)));
    connect(m_ui->customLabelsPlainTextEdit, SIGNAL(textChanged()), this,
            SLOT(onCustomLabelsListChanged()));

    // apply button
    connect(m_ui->applyPushButton, SIGNAL(clicked()), this, SLOT(onApply()));
    // close button
    connect(m_ui->closePushButton, SIGNAL(clicked()), this, SLOT(onClose()));

    // populate main combox box with all known scales
    updateMainComboBox();

    if (!m_colorScale)
        m_colorScale = m_manager->getDefaultScale(ccColorScalesManager::BGYR);

    setActiveScale(m_colorScale);
}

void ccColorScaleEditorDialog::setAssociatedScalarField(ccScalarField* sf) {
    m_associatedSF = sf;
    if (m_associatedSF &&
        (!m_colorScale ||
         m_colorScale->isRelative()))  // we only update those values if the
                                       // current scale is not absolute!
    {
        m_minAbsoluteVal = m_associatedSF->getMin();
        m_maxAbsoluteVal = m_associatedSF->getMax();
    }
}

void ccColorScaleEditorDialog::updateMainComboBox() {
    if (!m_manager) {
        assert(false);
        return;
    }

    m_ui->rampComboBox->blockSignals(true);
    m_ui->rampComboBox->clear();

    // populate combo box with scale names (and UUID)
    assert(m_manager);
    for (ccColorScalesManager::ScalesMap::const_iterator it =
                 m_manager->map().constBegin();
         it != m_manager->map().constEnd(); ++it)
        m_ui->rampComboBox->addItem((*it)->getName(), (*it)->getUuid());

    // find the currently selected scale in the new 'list'
    int pos = -1;
    if (m_colorScale) {
        pos = m_ui->rampComboBox->findData(m_colorScale->getUuid());
        if (pos < 0)  // the current color scale has disappeared?!
            m_colorScale = ccColorScale::Shared(nullptr);
    }
    m_ui->rampComboBox->setCurrentIndex(pos);

    m_ui->rampComboBox->blockSignals(false);
}

void ccColorScaleEditorDialog::colorScaleChanged(int pos) {
    QString UUID = m_ui->rampComboBox->itemData(pos).toString();
    ccColorScale::Shared colorScale =
            ccColorScalesManager::GetUniqueInstance()->getScale(UUID);

    setActiveScale(colorScale);
}

void ccColorScaleEditorDialog::relativeModeChanged(int value) {
    setScaleModeToRelative(value == 0 ? true : false);

    setModified(true);
}

void ccColorScaleEditorDialog::setModified(bool state) {
    m_modified = state;
    m_ui->saveToolButton->setEnabled(m_modified);
}

bool ccColorScaleEditorDialog::canChangeCurrentScale() {
    if (!m_colorScale || !m_modified) return true;

    if (m_colorScale->isLocked()) {
        assert(false);
        return true;
    }

    /// ask the user if we should save the current scale?
    QMessageBox::StandardButton button = QMessageBox::warning(
            this, "Current scale has been modified",
            "Do you want to save modifications?",
            QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel,
            QMessageBox::Cancel);
    if (button == QMessageBox::Yes) {
        if (!saveCurrentScale()) {
            return false;
        }
    } else if (button == QMessageBox::Cancel) {
        return false;
    }
    return true;
}

bool ccColorScaleEditorDialog::isRelativeMode() const {
    return (m_ui->scaleModeComboBox->currentIndex() == 0 ? true : false);
}

void ccColorScaleEditorDialog::setActiveScale(
        ccColorScale::Shared currentScale) {
    // the user wants to change the current scale while the it has been modified
    // (and potentially not saved)
    if (m_colorScale != currentScale) {
        if (!canChangeCurrentScale()) {
            // restore old combo-box state
            int pos = m_ui->rampComboBox->findData(m_colorScale->getUuid());
            if (pos >= 0) {
                m_ui->rampComboBox->blockSignals(true);
                m_ui->rampComboBox->setCurrentIndex(pos);
                m_ui->rampComboBox->blockSignals(false);
            } else {
                assert(false);
            }
            // stop process
            return;
        }
    }

    m_colorScale = currentScale;
    setModified(false);

    // make sure combo-box is up to date
    {
        int pos = m_ui->rampComboBox->findData(m_colorScale->getUuid());
        if (pos >= 0) {
            m_ui->rampComboBox->blockSignals(true);
            m_ui->rampComboBox->setCurrentIndex(pos);
            m_ui->rampComboBox->blockSignals(false);
        }
    }

    // setup dialog components
    {
        // locked state
        bool isLocked = !m_colorScale || m_colorScale->isLocked();
        m_ui->colorScaleParametersFrame->setEnabled(!isLocked);
        m_ui->exportToolButton->setEnabled(!isLocked);
        m_ui->lockWarningLabel->setVisible(isLocked);
        m_ui->selectedSliderGroupBox->setEnabled(!isLocked);
        m_scaleWidget->setEnabled(!isLocked);
        m_ui->customLabelsGroupBox->blockSignals(true);
        m_ui->customLabelsGroupBox->setEnabled(!isLocked);
        m_ui->customLabelsGroupBox->blockSignals(false);

        // absolute or relative mode
        if (m_colorScale) {
            bool isRelative = m_colorScale->isRelative();
            if (!isRelative) {
                // absolute color scales defines their own boundaries
                m_colorScale->getAbsoluteBoundaries(m_minAbsoluteVal,
                                                    m_maxAbsoluteVal);
            }
            setScaleModeToRelative(isRelative);
        } else {
            // shouldn't be accessible anyway....
            assert(isLocked == true);
            setScaleModeToRelative(false);
        }
    }

    // custom labels
    {
        ccColorScale::LabelSet& customLabels = m_colorScale->customLabels();
        if (customLabels.empty()) {
            m_ui->customLabelsPlainTextEdit->blockSignals(true);
            m_ui->customLabelsPlainTextEdit->setPlainText(
                    s_defaultEmptyCustomListText);
            m_ui->customLabelsPlainTextEdit->blockSignals(false);
        } else {
            QString text;
            size_t index = 0;
            for (ccColorScale::LabelSet::const_iterator it =
                         customLabels.begin();
                 it != customLabels.end(); ++it, ++index) {
                if (index != 0) text += QString("\n");
                text += QString::number(it->value, 'f', 6);
                if (!it->text.isEmpty()) {
                    text += " \"" + it->text + '\"';
                }
            }
            m_ui->customLabelsPlainTextEdit->blockSignals(true);
            m_ui->customLabelsPlainTextEdit->setPlainText(text);
            m_ui->customLabelsPlainTextEdit->blockSignals(false);
        }
        m_ui->customLabelsGroupBox->blockSignals(true);
        m_ui->customLabelsGroupBox->setChecked(!customLabels.empty());
        m_ui->customLabelsGroupBox->blockSignals(false);
    }

    m_scaleWidget->importColorScale(m_colorScale);

    onStepSelected(-1);
}

void ccColorScaleEditorDialog::setScaleModeToRelative(bool isRelative) {
    m_ui->scaleModeComboBox->setCurrentIndex(isRelative ? 0 : 1);
    m_ui->valueDoubleSpinBox->setSuffix(isRelative ? QString(" %") : QString());
    m_ui->valueDoubleSpinBox->blockSignals(true);
    if (isRelative)
        m_ui->valueDoubleSpinBox->setRange(0.0, 100.0);  // between 0 and 100%
    else
        m_ui->valueDoubleSpinBox->setRange(-1.0e9, 1.0e9);
    m_ui->valueDoubleSpinBox->blockSignals(false);

    // update selected slider frame
    int selectedIndex =
            (m_scaleWidget ? m_scaleWidget->getSelectedStepIndex() : -1);
    onStepModified(selectedIndex);
}

void ccColorScaleEditorDialog::onStepSelected(int index) {
    m_ui->selectedSliderGroupBox->setEnabled(
            /*m_colorScale && !m_colorScale->isLocked() && */ index >= 0);
    // don't delete the first and last steps!
    m_ui->deleteSliderToolButton->setEnabled(
            index >= 1 && index + 1 < m_scaleWidget->getStepCount());

    if (index < 0) {
        m_ui->valueDoubleSpinBox->blockSignals(true);
        m_ui->valueDoubleSpinBox->setValue(0.0);
        m_ui->valueDoubleSpinBox->blockSignals(false);
        ccQtHelpers::SetButtonColor(m_ui->colorToolButton, Qt::gray);
        m_ui->valueLabel->setVisible(false);
    } else {
        bool modified =
                m_modified;  // save 'modified' state before calling
                             // onStepModified (which will force it to true)
        onStepModified(index);
        setModified(modified);  // restore true 'modified' state
    }
}

void ccColorScaleEditorDialog::onStepModified(int index) {
    if (index < 0 || index >= m_scaleWidget->getStepCount()) return;

    const ColorScaleElementSlider* slider = m_scaleWidget->getStep(index);
    assert(slider);

    ccQtHelpers::SetButtonColor(m_ui->colorToolButton, slider->getColor());
    if (m_colorScale) {
        const double relativePos = slider->getRelativePos();
        if (isRelativeMode()) {
            m_ui->valueDoubleSpinBox->blockSignals(true);
            m_ui->valueDoubleSpinBox->setValue(relativePos * 100.0);
            m_ui->valueDoubleSpinBox->blockSignals(false);
            if (m_associatedSF) {
                // compute corresponding scalar value for associated SF
                double actualValue = m_associatedSF->getMin() +
                                     relativePos * (m_associatedSF->getMax() -
                                                    m_associatedSF->getMin());
                m_ui->valueLabel->setText(QString("(%1)").arg(actualValue));
                m_ui->valueLabel->setVisible(true);
            } else {
                m_ui->valueLabel->setVisible(false);
            }

            // can't change min and max boundaries in 'relative' mode!
            m_ui->valueDoubleSpinBox->setEnabled(
                    index > 0 && index < m_scaleWidget->getStepCount() - 1);
        } else {
            // compute corresponding 'absolute' value from current dialog
            // boundaries
            double absoluteValue =
                    m_minAbsoluteVal +
                    relativePos * (m_maxAbsoluteVal - m_minAbsoluteVal);

            m_ui->valueDoubleSpinBox->blockSignals(true);
            m_ui->valueDoubleSpinBox->setValue(absoluteValue);
            m_ui->valueDoubleSpinBox->blockSignals(false);
            m_ui->valueDoubleSpinBox->setEnabled(true);

            // display corresponding relative position as well
            m_ui->valueLabel->setText(
                    QString("(%1 %)").arg(relativePos * 100.0));
            m_ui->valueLabel->setVisible(true);
        }

        setModified(true);
    }
}

void ccColorScaleEditorDialog::deletecSelectedStep() {
    int selectedIndex = m_scaleWidget->getSelectedStepIndex();
    if (selectedIndex >= 1 &&
        selectedIndex + 1 <
                m_scaleWidget->getStepCount())  // never delete the first and
                                                // last steps!
    {
        m_scaleWidget->deleteStep(selectedIndex);
        setModified(true);
    }
}

void ccColorScaleEditorDialog::changeSelectedStepColor() {
    int selectedIndex = m_scaleWidget->getSelectedStepIndex();
    if (selectedIndex < 0) return;

    const ColorScaleElementSlider* slider =
            m_scaleWidget->getStep(selectedIndex);
    assert(slider);

    QColor newCol = QColorDialog::getColor(slider->getColor(), this);
    if (newCol.isValid()) {
        // eventually onStepModified will be called (and thus m_modified will be
        // updated)
        m_scaleWidget->setStepColor(selectedIndex, newCol);
    }
}

void ccColorScaleEditorDialog::changeSelectedStepValue(double value) {
    if (!m_scaleWidget) return;

    int selectedIndex = m_scaleWidget->getSelectedStepIndex();
    if (selectedIndex < 0) return;

    const ColorScaleElementSlider* slider =
            m_scaleWidget->getStep(selectedIndex);
    assert(slider);

    bool relativeMode = isRelativeMode();
    if (relativeMode) {
        assert(selectedIndex != 0 &&
               selectedIndex + 1 < m_scaleWidget->getStepCount());

        value /= 100.0;  // from percentage to relative position
        assert(value >= 0.0 && value <= 1.0);

        // eventually onStepModified will be called (and thus m_modified will be
        // updated)
        m_scaleWidget->setStepRelativePosition(selectedIndex, value);
    } else  // absolute scale mode
    {
        // we build up the new list based on absolute values
        SharedColorScaleElementSliders newSliders(
                new ColorScaleElementSliders());
        {
            for (int i = 0; i < m_scaleWidget->getStepCount(); ++i) {
                const ColorScaleElementSlider* slider =
                        m_scaleWidget->getStep(i);
                double absolutePos =
                        (i == selectedIndex
                                 ? value
                                 : m_minAbsoluteVal +
                                           slider->getRelativePos() *
                                                   (m_maxAbsoluteVal -
                                                    m_minAbsoluteVal));
                newSliders->push_back(new ColorScaleElementSlider(
                        absolutePos, slider->getColor()));
            }
        }

        // update min and max boundaries
        {
            newSliders->sort();
            m_minAbsoluteVal =
                    newSliders->front()->getRelativePos();  // absolute in fact!
            m_maxAbsoluteVal =
                    newSliders->back()->getRelativePos();  // absolute in fact!
        }

        // convert absolute pos to relative ones
        int newSelectedIndex = -1;
        {
            double range = std::max(m_maxAbsoluteVal - m_minAbsoluteVal, 1e-12);
            for (int i = 0; i < newSliders->size(); ++i) {
                double absoluteVal = newSliders->at(i)->getRelativePos();
                if (absoluteVal == value) newSelectedIndex = i;
                double relativePos = (absoluteVal - m_minAbsoluteVal) / range;
                newSliders->at(i)->setRelativePos(relativePos);
            }
        }

        // update the whole scale with new sliders
        m_scaleWidget->setSliders(newSliders);

        m_scaleWidget->setSelectedStepIndex(newSelectedIndex, true);

        setModified(true);
    }
}

bool ccColorScaleEditorDialog::exportCustomLabelsList(
        ccColorScale::LabelSet& labels) {
    assert(m_ui->customLabelsGroupBox->isChecked());
    labels.clear();

    QString text = m_ui->customLabelsPlainTextEdit->toPlainText();
    QStringList items =
            qtCompatSplitRegex(text, "\\s+", QtCompat::SkipEmptyParts);
    if (items.size() < 2) {
        assert(false);
        return false;
    }

    try {
        for (int i = 0; i < items.size(); ++i) {
            bool ok;
            double d = items[i].toDouble(&ok);
            if (!ok) {
                return false;
            }
            labels.insert(d);
        }
    } catch (const std::bad_alloc&) {
        CVLog::Error("Not enough memory to save the custom labels!");
        labels.clear();
        return false;
    }

    return true;
}

bool ccColorScaleEditorDialog::checkCustomLabelsList(bool showWarnings) {
    QString text = m_ui->customLabelsPlainTextEdit->toPlainText();
    QStringList items =
            qtCompatSplitRegex(text, "\\s+", QtCompat::SkipEmptyParts);
    if (items.size() < 2) {
        if (showWarnings)
            CVLog::Error("Not enough labels defined (2 at least are required)");
        return false;
    }

    for (int i = 0; i < items.size(); ++i) {
        bool ok;
        items[i].toDouble(&ok);
        if (!ok) {
            if (showWarnings)
                CVLog::Error(
                        QString("Invalid label value: '%1'").arg(items[i]));
            return false;
        }
    }

    return true;
}

void ccColorScaleEditorDialog::onCustomLabelsListChanged() {
    setModified(true);
}

void ccColorScaleEditorDialog::toggleCustomLabelsList(bool state) {
    // custom list enable
    if (state) {
        QString previousText = m_ui->customLabelsPlainTextEdit->toPlainText();
        // if the previous list was 'empty', we clear its (fake) content
        if (previousText == s_defaultEmptyCustomListText) {
            m_ui->customLabelsPlainTextEdit->blockSignals(true);
            m_ui->customLabelsPlainTextEdit->clear();
            m_ui->customLabelsPlainTextEdit->blockSignals(false);
        }
    } else {
        if (!checkCustomLabelsList(false)) {
            // if the text is invalid
            m_ui->customLabelsPlainTextEdit->setPlainText(
                    s_defaultEmptyCustomListText);
        }
    }
    setModified(true);
}

void ccColorScaleEditorDialog::copyCurrentScale() {
    if (!m_colorScale) {
        assert(false);
        return;
    }

    ccColorScale::Shared scale =
            ccColorScale::Create(m_colorScale->getName() + QString("_copy"));
    if (!m_colorScale->isRelative()) {
        double minVal, maxVal;
        m_colorScale->getAbsoluteBoundaries(minVal, maxVal);
        scale->setAbsolute(minVal, maxVal);
    }
    m_scaleWidget->exportColorScale(scale);

    assert(m_manager);
    if (m_manager) m_manager->addScale(scale);

    updateMainComboBox();

    setActiveScale(scale);
}

bool ccColorScaleEditorDialog::saveCurrentScale() {
    if (!m_colorScale || m_colorScale->isLocked()) {
        assert(false);
        return false;
    }

    // check the custom labels
    if (m_ui->customLabelsGroupBox->isChecked() &&
        !checkCustomLabelsList(true)) {
        // error message already issued
        return false;
    }

    m_scaleWidget->exportColorScale(m_colorScale);
    bool wasRelative = m_colorScale->isRelative();
    bool isRelative = isRelativeMode();
    if (isRelative)
        m_colorScale->setRelative();
    else
        m_colorScale->setAbsolute(m_minAbsoluteVal, m_maxAbsoluteVal);

    // DGM: warning, if the relative state has changed
    // we must update all the SFs currently relying on this scale!
    if ((!isRelative || isRelative != wasRelative) && m_mainApp &&
        m_mainApp->dbRootObject()) {
        ccHObject::Container clouds;
        m_mainApp->dbRootObject()->filterChildren(clouds, true,
                                                  CV_TYPES::POINT_CLOUD, true);
        for (size_t i = 0; i < clouds.size(); ++i) {
            ccPointCloud* cloud = static_cast<ccPointCloud*>(clouds[i]);
            for (unsigned j = 0; j < cloud->getNumberOfScalarFields(); ++j) {
                ccScalarField* sf =
                        static_cast<ccScalarField*>(cloud->getScalarField(j));
                if (sf->getColorScale() == m_colorScale) {
                    // trick: we unlink then re-link the color scale to update
                    // everything automatically
                    sf->setColorScale(ccColorScale::Shared(nullptr));
                    sf->setColorScale(m_colorScale);

                    if (cloud->getCurrentDisplayedScalarField() == sf) {
                        // cloud->prepareDisplayForRefresh();
                        if (cloud->getParent() &&
                            cloud->getParent()->isKindOf(CV_TYPES::MESH)) {
                            // for mesh vertices (just in case)
                            // cloud->getParent()->prepareDisplayForRefresh();
                        }
                    }
                }
            }
        }

        m_mainApp->refreshAll();
    }

    // save the custom labels
    if (m_ui->customLabelsGroupBox->isChecked()) {
        exportCustomLabelsList(m_colorScale->customLabels());
    } else {
        m_colorScale->customLabels().clear();
    }

    setModified(false);

    return true;
}

void ccColorScaleEditorDialog::renameCurrentScale() {
    if (!m_colorScale || m_colorScale->isLocked()) {
        assert(false);
        return;
    }

    QString newName =
            QInputDialog::getText(this, "Scale name", "Name", QLineEdit::Normal,
                                  m_colorScale->getName());
    if (!newName.isNull()) {
        m_colorScale->setName(newName);
        // position in combo box
        int pos = m_ui->rampComboBox->findData(m_colorScale->getUuid());
        if (pos >= 0)
            // update combo box entry name
            m_ui->rampComboBox->setItemText(pos, newName);
    }
}

void ccColorScaleEditorDialog::deleteCurrentScale() {
    if (!m_colorScale || m_colorScale->isLocked()) {
        assert(false);
        return;
    }

    // ask for confirmation
    if (QMessageBox::warning(this, "Delete scale", "Are you sure?",
                             QMessageBox::Yes | QMessageBox::No,
                             QMessageBox::No) == QMessageBox::No) {
        return;
    }

    // backup current scale
    ccColorScale::Shared colorScaleToDelete = m_colorScale;
    setModified(false);  // cancel any modification

    int currentIndex = m_ui->rampComboBox->currentIndex();
    if (currentIndex == 0)
        currentIndex = 1;
    else if (currentIndex > 0)
        --currentIndex;

    assert(m_manager);
    if (m_manager) {
        // activate the neighbor scale in the list
        ccColorScale::Shared nextScale = m_manager->getScale(
                m_ui->rampComboBox->itemData(currentIndex).toString());
        setActiveScale(nextScale);

        m_manager->removeScale(colorScaleToDelete->getUuid());
    }

    updateMainComboBox();
}

void ccColorScaleEditorDialog::createNewScale() {
    ccColorScale::Shared scale = ccColorScale::Create("New scale");

    // add default min and max steps
    scale->insert(ccColorScaleElement(0.0, Qt::blue), false);
    scale->insert(ccColorScaleElement(1.0, Qt::red), true);

    assert(m_manager);
    if (m_manager) m_manager->addScale(scale);

    updateMainComboBox();

    setActiveScale(scale);
}

void ccColorScaleEditorDialog::onApply() {
    if (m_mainApp && canChangeCurrentScale()) {
        if (m_associatedSF) m_associatedSF->setColorScale(m_colorScale);
        m_mainApp->refreshAll();
    }
}

void ccColorScaleEditorDialog::onClose() {
    if (canChangeCurrentScale()) {
        accept();
    }
}

void ccColorScaleEditorDialog::exportCurrentScale() {
    if (!m_colorScale || m_colorScale->isLocked()) {
        assert(false);
        return;
    }

    // persistent settings
    QSettings settings;
    settings.beginGroup(ecvPS::SaveFile());
    QString currentPath =
            settings.value(ecvPS::CurrentPath(), ecvFileUtils::defaultDocPath())
                    .toString();

    // ask for a filename
    QString filename = QFileDialog::getSaveFileName(this, "Select output file",
                                                    currentPath, "*.xml");
    if (filename.isEmpty()) {
        // process cancelled by user
        return;
    }

    // save last saving location
    settings.setValue(ecvPS::CurrentPath(), QFileInfo(filename).absolutePath());
    settings.endGroup();

    // try to save the file
    if (m_colorScale->saveAsXML(filename)) {
        CVLog::Print(
                QString("[ColorScale] Scale '%1' successfully exported in '%2'")
                        .arg(m_colorScale->getName(), filename));
    }
}

void ccColorScaleEditorDialog::importScale() {
    // persistent settings
    QSettings settings;
    settings.beginGroup(ecvPS::LoadFile());
    QString currentPath =
            settings.value(ecvPS::CurrentPath(), ecvFileUtils::defaultDocPath())
                    .toString();

    // ask for a filename
    QString filename = QFileDialog::getOpenFileName(
            this, "Select color scale file", currentPath, "*.xml");
    if (filename.isEmpty()) {
        // process cancelled by user
        return;
    }

    // save last loading parameters
    settings.setValue(ecvPS::CurrentPath(), QFileInfo(filename).absolutePath());
    settings.endGroup();

    // try to load the file
    ccColorScale::Shared scale = ccColorScale::LoadFromXML(filename);
    if (scale) {
        assert(m_manager);
        if (m_manager) {
            ccColorScale::Shared otherScale =
                    m_manager->getScale(scale->getUuid());
            if (otherScale) {
                QString message = "A color scale with the same UUID";
                if (otherScale->getName() == scale->getName())
                    message += QString(" and the same name (%1)")
                                       .arg(scale->getName());
                message += " is already in store!";
                message += "\n";
                message +=
                        "Do you want to force the importation of this new "
                        "scale? (a new UUID will be generated)";

                if (QMessageBox::question(this, "UUID conflict", message,
                                          QMessageBox::Yes,
                                          QMessageBox::No) == QMessageBox::No) {
                    CVLog::Warning(
                            "[ccColorScaleEditorDialog::importScale] "
                            "Importation cancelled due to a conflicting UUID "
                            "(color scale may already be in store)");
                    return;
                }
                // generate a new UUID
                scale->setUuid(QUuid::createUuid().toString());
            }
            // now we can import the scale
            m_manager->addScale(scale);
            CVLog::Print(QString("[ccColorScaleEditorDialog::importScale] "
                                 "Color scale '%1' successfully imported")
                                 .arg(scale->getName()));
        }

        updateMainComboBox();

        setActiveScale(scale);
    }
}
