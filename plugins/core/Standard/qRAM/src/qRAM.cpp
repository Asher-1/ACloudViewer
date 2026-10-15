// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qRAM.h"

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QAction>
#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMainWindow>
#include <QSet>

#include "RAMDialog.h"
qRAM::qRAM(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qRAM/info.json") {
    m_action = new QAction(QIcon(":/CC/plugin/qRAM/images/qRAM.svg"),
                           tr("Recognize Anything (RAM++)"), this);
    connect(m_action, &QAction::triggered, this, &qRAM::showDialog);
    connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit, this,
            [this]() {
                if (m_dialog) m_dialog->shutdown();
            });
}
qRAM::~qRAM() { delete m_dialog; }
QList<QAction*> qRAM::getActions() { return {m_action}; }
void qRAM::onNewSelection(const ccHObject::Container& selected) {
    m_selection = selected;
    refreshSelection();
}
void qRAM::refreshSelection() {
    if (!m_dialog) return;
    QList<QPair<QString, QImage>> images;
    ccHObject::Container candidates = m_selection;
    for (auto* obj : m_selection)
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
void qRAM::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new RAMDialog(m_app->getMainWindow());
        connect(m_dialog, &RAMDialog::addToDbRequested, this,
                [this](const RAMResult& result) {
                    const QString name = ecvPluginDbNaming::makeUnique(
                            QStringLiteral("RAM_%1_%2_%3")
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
                    auto* image = new ccImage(result.annotated.isNull()
                                                      ? result.image
                                                      : result.annotated,
                                              name);
                    const auto timings =
                            result.json.value("timings").toObject();
                    QStringList selectedTags;
                    for (const auto& value :
                         result.json.value("selected_tags").toArray())
                        selectedTags << value.toString();
                    image->setMetaData("RAM", true);
                    image->setMetaData("Model",
                                       result.json["model"].toString());
                    image->setMetaData("Device", result.device);
                    image->setMetaData("Source", result.source);
                    image->setMetaData("Runtime (ms)", result.e2eMs);
                    image->setMetaData("RAM/Selected tags",
                                       selectedTags.join('\n'));
                    image->setMetaData("RAM/Caption",
                                       result.json.value("caption").toString());
                    image->setMetaData(
                            "RAM/Preprocess (ms)",
                            timings.value("preprocess_ms").toDouble());
                    image->setMetaData(
                            "RAM/Inference (ms)",
                            timings.value("inference_ms").toDouble());
                    image->setMetaData(
                            "RAM/Postprocess (ms)",
                            timings.value("postprocess_ms").toDouble());
                    image->setMetaData(
                            "RAM/Context init (ms)",
                            timings.value("context_init_ms").toDouble());
                    image->setMetaData(
                            "RAM/Cold start",
                            !timings.value("context_reused").toBool());
                    image->setMetaData(
                            "RAM/Result",
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
