// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qRelateAnything.h"

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QAction>
#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMainWindow>
#include <QPainter>
#include <QSet>

#include "RelateAnythingDialog.h"
qRelateAnything::qRelateAnything(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qRelateAnything/info.json") {
    m_action = new QAction(
            QIcon(":/CC/plugin/qRelateAnything/images/qRelateAnything.svg"),
            tr("RelateAnything"), this);
    connect(m_action, &QAction::triggered, this, &qRelateAnything::showDialog);
    connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit, this,
            [this]() {
                if (m_dialog) m_dialog->shutdown();
            });
}
qRelateAnything::~qRelateAnything() { delete m_dialog; }
QList<QAction*> qRelateAnything::getActions() { return {m_action}; }
void qRelateAnything::onNewSelection(const ccHObject::Container& selected) {
    m_selection = selected;
    refreshSelection();
}
void qRelateAnything::refreshSelection() {
    if (!m_dialog || !m_app) return;
    // M7 fix: query current selection from app instead of using stored
    // raw pointers, which may have become dangling if objects were deleted.
    const ccHObject::Container& selected = m_app->getSelectedEntities();
    QList<QPair<QString, QImage>> images;
    ccHObject::Container candidates = selected;
    for (auto* obj : selected)
        if (obj && obj->isGroup())
            obj->filterChildren(candidates, true, CV_TYPES::IMAGE, false);
    QSet<unsigned> ids;
    for (auto* obj : candidates) {
        auto* image = dynamic_cast<ccImage*>(obj);
        if (image && !image->data().isNull() &&
            !ids.contains(obj->getUniqueID())) {
            images.append({obj->getName(), image->data()});
            ids.insert(obj->getUniqueID());
        }
    }
    m_dialog->setDbImages(images);
}
void qRelateAnything::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new RelateAnythingDialog(m_app->getMainWindow());
        connect(m_dialog, &RelateAnythingDialog::addToDbRequested, this,
                [this](const RelateAnythingResult& result) {
                    const QString name = ecvPluginDbNaming::makeUnique(
                            QStringLiteral("RelateAnything_%1_%2_%3")
                                    .arg(ecvPluginDbNaming::
                                                 modelTagFromFilename(
                                                         result.json["model"]
                                                                 .toString()),
                                         ecvPluginDbNaming::sanitizeSegment(
                                                 result.source.section('/',
                                                                       -1)),
                                         ecvPluginDbNaming::deviceTagFromName(
                                                 result.device)),
                            m_app);
                    auto* image = new ccImage(result.annotated, name);
                    image->setMetaData("RelateAnything", true);
                    image->setMetaData("Model",
                                       result.json["model"].toString());
                    image->setMetaData("Device", result.device);
                    image->setMetaData("Source", result.source);
                    image->setMetaData("Runtime (ms)", result.e2eMs);
                    image->setMetaData(
                            "RelateAnything/Result",
                            QString::fromUtf8(
                                    QJsonDocument(result.json)
                                            .toJson(QJsonDocument::Compact)));
                    m_app->addToDB(image, true, true, false, true);
                    m_app->setSelectedInDB(image, true);
                });
    }
    refreshSelection();
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}
