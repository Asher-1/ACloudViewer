// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvTextureFileSelector.h"

// Qt
#include <QComboBox>
#include <QHBoxLayout>
#include <QToolButton>

ecvTextureFileSelector::ecvTextureFileSelector(
        QWidget* parent, QString defaultButtonIconPath /*=QString()*/)
    : QFrame(parent), m_comboBox(new QComboBox()), m_button(new QToolButton()) {
    setLayout(new QHBoxLayout());
    layout()->setContentsMargins(0, 0, 0, 0);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

    // combox box
    if (m_comboBox) {
        layout()->addWidget(m_comboBox);
    }

    // tool button
    if (m_button) {
        m_button->setIcon(QIcon(defaultButtonIconPath));
        layout()->addWidget(m_button);
    }
}

bool ecvTextureFileSelector::isEmpty() const {
    return !m_comboBox || m_comboBox->count() <= 0;
}

void ecvTextureFileSelector::init(
        const QMap<QString, QString>& texturePathMap) {
    // fill combox box
    if (m_comboBox) {
        m_comboBox->disconnect(this);

        m_comboBox->clear();

        for (QMap<QString, QString>::const_iterator path =
                     texturePathMap.constBegin();
             path != texturePathMap.constEnd(); ++path) {
            m_comboBox->addItem(path.key(), path.value());
        }

        connect(m_comboBox, SIGNAL(activated(int)), this,
                SIGNAL(textureFileSelected(int)));
    }
    // advanced tool button
    if (m_button) {
        m_button->disconnect(this);
        connect(m_button, SIGNAL(clicked()), this,
                SIGNAL(textureFileEditorSummoned()));
    }
}

void ecvTextureFileSelector::addItem(const QString& textureFilename,
                                     const QString& textureFilepath) {
    if (m_comboBox && m_comboBox->findData(textureFilepath) < 0) {
        m_comboBox->addItem(textureFilename, textureFilepath);
        setSelectedTexturefile(textureFilepath);
    }
}

QString ecvTextureFileSelector::getTexturefilePath(int index) const {
    if (!m_comboBox || index < 0 || index >= m_comboBox->count())
        return QString();

    // get UUID associated to the combo-box item
    return m_comboBox->itemData(index).toString();
}

void ecvTextureFileSelector::setSelectedTexturefile(QString textureFilepath) {
    if (!m_comboBox) return;

    // search right index by UUID
    int pos = m_comboBox->findData(textureFilepath);
    if (pos < 0) return;
    m_comboBox->setCurrentIndex(pos);

    emit textureFileSelected(pos);
}
