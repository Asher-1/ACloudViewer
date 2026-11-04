// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qVLXSeek.h"

#include <aicore/inference_log.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>

#include <QDir>
#include <QFileInfo>
#include <QRegExp>

#include "VLXSeekBatchCommand.h"
#include "ecvAICoreRuntimeHelpers.h"
#include "ecvPluginDbNaming.h"

namespace {
// Strip VLX-Seek internal XML tags from the answer for human-readable
// display. The raw model output uses:
//   <ground>text</ground><objects><obj1><obj2></objects>  (detection)
//   <think>...</think><answer>text</answer>               (reasoning)
//   plain text                                             (vqa)
//
// This function:
// - Preserves <think>...</think> blocks (reasoning process visibility)
// - Extracts text from <answer>...</answer> if present
// - Strips <ground>, <objects>, <objN> tags but preserves text content
// - Converts <objects><obj1><obj2></objects> to [obj1, obj2] for context
QString stripAnswerTags(const QString& answer) {
    if (answer.isEmpty()) return answer;
    QString result = answer;
    // Extract text from <answer>...</answer> if present (reasoning tasks)
    QRegExp answerRe(QStringLiteral("<answer>\\s*(.*)\\s*</answer>"),
                     Qt::CaseSensitive, QRegExp::RegExp2);
    if (answerRe.indexIn(result) >= 0) {
        // Keep <think>...</think> + extracted answer
        QString thinkBlock;
        QRegExp thinkRe(QStringLiteral("<think>(.*)</think>"),
                        Qt::CaseSensitive, QRegExp::RegExp2);
        if (thinkRe.indexIn(result) >= 0) {
            thinkBlock = QStringLiteral("<think>") + thinkRe.cap(1) +
                         QStringLiteral("</think>\n\n");
        }
        result = thinkBlock + answerRe.cap(1);
    } else {
        // No <answer> tag; remove <think>...</think> if present (shouldn't
        // happen, but handle gracefully)
        result.remove(QRegExp(QStringLiteral("<think>.*</think>"),
                              Qt::CaseSensitive, QRegExp::RegExp2));
    }
    // Replace <ground>text</ground> with just the text
    result.remove(QRegExp(QStringLiteral("</?ground>")));
    // Replace <objects>...</objects> with [obj1, obj2, ...]
    QRegExp objectsRe(QStringLiteral("<objects>(.*)</objects>"),
                      Qt::CaseSensitive, QRegExp::RegExp2);
    if (objectsRe.indexIn(result) >= 0) {
        QString objs = objectsRe.cap(1);
        QStringList objList;
        QRegExp objRe(QStringLiteral("<obj(\\d+)>"), Qt::CaseSensitive,
                      QRegExp::RegExp2);
        int pos = 0;
        while ((pos = objRe.indexIn(objs, pos)) != -1) {
            objList << QStringLiteral("obj%1").arg(objRe.cap(1));
            pos += objRe.matchedLength();
        }
        if (objList.count() > 0) {
            result = result.replace(objectsRe.cap(0),
                                    QStringLiteral(" [%1]").arg(objList.join(
                                            QStringLiteral(", "))));
        } else {
            result.remove(objectsRe.cap(0));
        }
    }
    // Clean up any remaining tags
    result.remove(QRegExp(QStringLiteral("</?[a-z]+>")));
    return result.trimmed();
}
}  // namespace

qVLXSeek::qVLXSeek(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qVLXSeek/info.json") {}

void qVLXSeek::onNewSelection(const ccHObject::Container& selectedEntities) {
    if (!m_dialog) return;
    for (ccHObject* entity : selectedEntities) {
        auto* img = ccHObjectCaster::ToImage(entity);
        if (!img) continue;
        // Result images carry their source path as metadata; prefer that,
        // otherwise fall back to the image's file path when one was set.
        QVariant source = img->getMetaData(QStringLiteral("Source"));
        if (source.isValid() && !source.toString().isEmpty() &&
            QFileInfo::exists(source.toString())) {
            m_dialog->setInputPath(source.toString());
            return;
        }
    }
}

QList<QAction*> qVLXSeek::getActions() {
    if (!m_action) {
        m_action =
                new QAction(QIcon(":/CC/plugin/qVLXSeek/images/qVLXSeek.svg"),
                            tr("VLX-Seek grounding"), this);
        m_action->setToolTip(
                tr("VLX-Seek fine-grained open-vocabulary grounding: "
                   "detection, counting, region caption and region OCR via "
                   "region-reference retrieval"));
        connect(m_action, &QAction::triggered, this, &qVLXSeek::showDialog);
    }
    return {m_action};
}

void qVLXSeek::registerCommands(ccCommandLineInterface* cmd) {
    if (!cmd) return;
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandVlxSeekRun()));
}

void qVLXSeek::showDialog() {
    if (!m_dialog) {
        // The worker thread hands results back through queued connections;
        // without runtime registration Qt silently drops the struct-carrying
        // signal and onResultReady never runs (no DB export, no summary).
        qRegisterMetaType<VLXSeekRunResult>("VLXSeekRunResult");
        qRegisterMetaType<VLXSeekWorker::Settings>("VLXSeekWorker::Settings");
        m_dialog = new VLXSeekDialog(nullptr);
        connect(m_dialog, &VLXSeekDialog::runRequested, this,
                [this](const VLXSeekWorker::Settings& settings) {
                    m_lastSettings = settings;
                });
        connect(m_dialog, &VLXSeekDialog::resultReady, this,
                [this](const VLXSeekRunResult& result) {
                    onResultReady(result);
                });
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qVLXSeek::onResultReady(const VLXSeekRunResult& result) {
    // VQA produces a text answer only (no boxes, no visual change to the
    // input image); exporting an unmodified image to the DB tree adds noise
    // without information. The full Q&A is already in the console log below.
    // All other tasks produce annotated images worth keeping in the DB tree.
    const bool isVqa = m_lastSettings.task == QStringLiteral("vqa");
    if (!isVqa && m_dialog && m_dialog->exportToDbRequested()) {
        addResultToDb(result, m_lastSettings);
    }

    // One line with every user-visible key fact: task mode, input, proposal
    // supply (source + count — wedetect, external JSON or ocr-det text
    // boxes), detection count, the staged timings (load / inference / e2e)
    // and the human-readable answer. The task mode is the first field so that
    // scanning the log immediately identifies which of the seven supported
    // tasks (detection / grounding_single / counting / reasoning_detection /
    // brief_region_caption / region_ocr / vqa) produced each row.
    // stripAnswerTags preserves <think>...</think> reasoning steps while
    // stripping <ground>, <objects>, <objN> tags for readability.
    const QString readableAnswer = stripAnswerTags(result.answer);
    // The ocr-det proposer runs plugin-side (context load + det + rec); its
    // wall time is not part of the VLX-Seek timing contract, so it is
    // reported next to the source instead of the staged timings.
    const QString proposalDetail =
            result.proposalMs > 0.0 ? QStringLiteral(" (proposer %1 ms)")
                                              .arg(result.proposalMs, 0, 'f', 0)
                                    : QString();
    aicore_inference_log::log(
            QStringLiteral("[qVLXSeek] task=%1 | %2\n"
                           "           %3 box(es) from %4 %5 proposals%6 | "
                           "load %7 ms, infer %8 ms, e2e %9 ms | "
                           "device %10 (%11)\n"
                           "           answer: %12")
                    .arg(m_lastSettings.task, result.imagePath)
                    .arg(result.boxes.size())
                    .arg(result.proposalCount)
                    .arg(result.proposalSource)
                    .arg(proposalDetail)
                    .arg(result.modelLoadMs, 0, 'f', 0)
                    .arg(result.inferenceMs, 0, 'f', 0)
                    .arg(result.e2eMs, 0, 'f', 0)
                    .arg(result.resolvedDevice.isEmpty()
                                 ? m_lastSettings.device
                                 : result.resolvedDevice)
                    .arg(result.precision.isEmpty() ? QStringLiteral("auto")
                                                    : result.precision)
                    .arg(readableAnswer));
}

void qVLXSeek::addResultToDb(const VLXSeekRunResult& result,
                             const VLXSeekWorker::Settings& settings) {
    if (!m_app || result.annotatedImage.isNull()) return;

    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(settings.visionPath);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("VLXSeek_%1_%2")
                    .arg(modelTag, QFileInfo(result.imagePath).baseName()),
            m_app);
    auto* img = new ccImage(result.annotatedImage, name);
    img->setMetaData(QStringLiteral("VLXSeek"), true);
    img->setMetaData(QStringLiteral("VLXSeek/Task"), settings.task);
    img->setMetaData(QStringLiteral("VLXSeek/Text"), settings.text);
    img->setMetaData(QStringLiteral("VLXSeek/Answer"), result.answer);
    img->setMetaData(QStringLiteral("VLXSeek/Boxes"),
                     static_cast<qlonglong>(result.boxes.size()));
    img->setMetaData(QStringLiteral("VLXSeek/Proposals"),
                     static_cast<qlonglong>(result.proposalCount));
    img->setMetaData(QStringLiteral("VLXSeek/ProposalSource"),
                     result.proposalSource);
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.e2eMs);
    img->setMetaData(QStringLiteral("Source"), result.imagePath);
    if (!result.resolvedDevice.isEmpty())
        img->setMetaData(QStringLiteral("Device"), result.resolvedDevice);
    if (!result.precision.isEmpty())
        img->setMetaData(QStringLiteral("Precision"), result.precision);

    m_app->addToDB(img, true, true, false, true);
    // Select the result so it is immediately visible in the DB tree
    // (the shared AICore plugin finishing touch, qRMBG/qMapAnything
    // pattern).
    m_app->setSelectedInDB(img, true);
}
